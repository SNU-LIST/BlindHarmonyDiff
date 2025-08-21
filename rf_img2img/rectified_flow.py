from __future__ import annotations

import math
from collections import namedtuple
from typing import Tuple, Literal, Callable

import torch
from torch import Tensor
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchdiffeq import odeint

from torchvision.utils import save_image
import nibabel as nib

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from pathlib import Path
from torch.utils.data import Dataset
from functools import partial

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from torch.amp import autocast, GradScaler

from tensorboardX import SummaryWriter
import logging
import os
from datetime import datetime
#from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from .metrics import psnr, ssim, ssim3D

import torch
import nibabel as nib
import numpy as np
import os

def save_nifti(volume, output_path, voxel_size=(1.2, 1.2, 1.25), origin=(0, 0, 0)):
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]  # X축 해상도
    affine[1, 1] = voxel_size[1]  # Y축 해상도
    affine[2, 2] = voxel_size[2]  # Z축 해상도
    affine[:3, 3] = origin  # 좌표계 원점 설정

    for i in range(volume.shape[0]):
        volume_3d = volume[i, 0, :, :, :].cpu().numpy()
        nifti_img = nib.Nifti1Image(volume_3d, affine=affine)
        nib.save(nifti_img, output_path)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

# losses

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow',
        loss_fn: Literal[
            'mse',
            'pseudo_huber',
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: Tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        clip_during_sampling = False,
        clip_values: Tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: Tuple[float, float] = (-3., 3)
    ):
        super().__init__()

        if isinstance(model, dict):
            model = Unet(**model)

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)
        self.predict = predict

        # automatically default to a working setting for predict epsilon
        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn

        if loss_fn == 'mse':
            loss_fn = MSELoss()

        elif loss_fn == 'pseudo_huber':
            assert predict == 'flow'

            # section 4.2 of https://arxiv.org/abs/2405.20320v1
            loss_fn = PseudoHuberLoss(**loss_fn_kwargs)

        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules

        if noise_schedule == 'cosmap':
            noise_schedule = cosmap

        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction

        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency flow matching

        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        # immiscible diffusion paper, will be removed if does not work

        self.immiscible = immiscible

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: Module, noised, *, times, eps = 1e-10, data_init = None):
        """
        returns the model output as well as the derived flow, depending on the `predict` objective
        """

        batch = noised.shape[0]

        # prepare maybe time conditioning for model

        model_kwargs = dict()
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')

            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)

            model_kwargs.update(**{time_kwarg: times})

        if data_init is not None:
            output = model(torch.cat([noised, data_init.to(noised.device)], dim=1), **model_kwargs)
        else:
            output = model(noised, **model_kwargs)

        # depending on objective, derive flow

        if self.predict == 'flow':
            flow = output

        elif self.predict == 'noise':
            noise = output
            padded_times = append_dims(times, noised.ndim - 1)

            flow = (noised - noise) / padded_times.clamp(min = eps)

        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow

    @torch.no_grad()
    def sample_tra(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        data_init = None, # hy
        data_sc = None, # hy
        alpha = 0.02,
        num = 16,
        data_shape: Tuple[int, ...] | None = None,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity

        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function

        def ode_fn(t, x):
            x = maybe_clip(x)

            _, flow = self.predict_flow(model, x, times = t, data_init=data_init, **model_kwargs)     

            flow = maybe_clip_flow(flow)


            return flow

        # start with random gaussian noise - y0

        noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode
        step_div = np.round(np.arange(0, steps, steps/num) ).astype('int')
        sampled_data = noise
        for i in range(0,num):
            if i < num -1 :
                trajectory = odeint(ode_fn, sampled_data, times[step_div[i]:step_div[i+1]+1], **self.odeint_kwargs)
                corr_target = data_sc * times[step_div[i+1]] + noise * (1-times[step_div[i+1]])
            else:
                trajectory = odeint(ode_fn, sampled_data, times[step_div[i]:steps], **self.odeint_kwargs)
                corr_target = data_sc * times[-1] + noise * (1-times[-1])


            sampled_data = trajectory[-1]
            corr_target = corr_target - corr_target.mean()
            sampled_data_mean = sampled_data.mean()
            sampled_data_upd = sampled_data- sampled_data_mean
            grad = -corr_target/(1e-6+torch.sqrt(torch.mean(sampled_data_upd**2)))/(1e-6+torch.sqrt(torch.mean(corr_target**2))) \
            + torch.mean(sampled_data_upd*corr_target)*sampled_data_upd/(1e-6+torch.sqrt(torch.mean(sampled_data_upd**2))**3)/(1e-6+torch.sqrt(torch.mean(corr_target**2)))      
            sampled_data = sampled_data_upd - alpha*grad + sampled_data_mean

        self.train(was_training)

        return sampled_data


    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        data_init = None, # hy
        data_shape: Tuple[int, ...] | None = None,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick
        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function
        def ode_fn(t, x, data_init=None):
            x = maybe_clip(x)

            _, flow = self.predict_flow(model, x, times = t, data_init=data_init, **model_kwargs)

            flow = maybe_clip_flow(flow)

            return flow

        # start with random gaussian noise - y0
        noise = torch.randn((batch_size, *data_shape), device = self.device)

        # time steps
        times = torch.linspace(0., 1., steps, device = self.device)

        # ode
        trajectory = odeint(lambda t, x: ode_fn(t, x, data_init), noise, times, **self.odeint_kwargs) #odeint(ode_fn, noise, times, **self.odeint_kwargs)

        # sampled_data = trajectory[-1]

        self.train(was_training)

        return trajectory #sampled_data

    def forward(
        self,
        data,
        data_init: Tensor | None = None,
        return_loss_breakdown = False,
    ):
        batch, *data_shape = data.shape

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data
        noise = torch.randn_like(data)

        # maybe immiscible flow

        if self.immiscible:
            cost = torch.cdist(data.flatten(1), noise.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.cpu())
            noise = noise[from_numpy(reorder_indices).to(cost.device)]

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        # time needs to be from [0, 1 - delta_time] if using consistency loss

        if self.use_consistency:
            padded_times *= 1. - self.consistency_delta_time

        def get_noised_and_flows(model, t):

            # maybe noise schedule

            t = self.noise_schedule(t)

            # Algorithm 2 in paper
            # linear interpolation of noise with data using random times
            # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

            noised = t * data + (1. - t) * noise

            # the model predicts the flow from the noised data

            flow = data - noise

            model_output, pred_flow = self.predict_flow(model, noised, times=t, data_init=data_init)

            # predicted data will be the noised xt + flow * (1. - t)

            pred_data = noised + pred_flow * (1. - t)

            return model_output, flow, pred_flow, pred_data

        # getting flow and pred flow for main model

        output, flow, pred_flow, pred_data = get_noised_and_flows(self.model, padded_times)

        # if using consistency loss, also need the ema model predicted flow

        if self.use_consistency:
            delta_t = self.consistency_delta_time
            ema_output, ema_flow, ema_pred_flow, ema_pred_data = get_noised_and_flows(self.ema_model, padded_times + delta_t)

        # determine target, depending on objective

        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')

        # losses

        main_loss = self.loss_fn(output, target, pred_data = pred_data, times = times, data = data)

        consistency_loss = data_match_loss = velocity_match_loss = 0.

        if self.use_consistency:
            # consistency losses from consistency fm paper - eq (6) in https://arxiv.org/html/2407.02398v1

            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)

            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha

        # total loss

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss

        # loss breakdown

        return total_loss, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)

# unet

def conv_nd(dims, in_channels, out_channels, kernel_size, **kwargs):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0


def Upsample(dim, dim_out=None, dims=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv_nd(dims, dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None, dims=2):
    return nn.Sequential(
        Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=2, p2=2, p3=2) if dims == 3 else 
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        conv_nd(dims, dim * (8 if dims == 3 else 4), default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim, dims=2):
        super().__init__()
        self.scale = dim ** 0.5
        if dims == 2:
            self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        elif dims == 3:
            self.gamma = nn.Parameter(torch.zeros(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * (self.gamma + 1) * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0., dims=2):
        super().__init__()
        self.proj = conv_nd(dims, dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out, dims=dims)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0., dims=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout, dims=dims)
        self.block2 = Block(dim_out, dim_out, dims=dims)
        self.res_conv = conv_nd(dims, dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.dims = dims

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            if self.dims == 2:
                time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            elif self.dims == 3:
                time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        dims = 2
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, dims=dims)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = conv_nd(dims, dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            conv_nd(dims, hidden_dim, dim, 1),
            RMSNorm(dim, dims=dims)
        )

        self.dims = dims

    def forward(self, x):
        if self.dims == 2:
            b, c, h, w = x.shape
        elif self.dims == 3:
            b, c, h, w, d = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        if self.dims == 2:
            q, k, v = tuple(rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads) for t in qkv)
        elif self.dims == 3:
            q, k, v = tuple(rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads) for t in qkv) # 수정

        mk, mv = tuple(repeat(t, 'h c n -> b h c n', b = b) for t in self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = einsum(k, v, 'b h d n, b h e n -> b h d e')
        out = einsum(context, q, 'b h d e, b h d n -> b h e n')

        if self.dims == 2:
            out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        elif self.dims == 3:
            out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h=self.heads, x=h, y=w, z=d)

        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False,
        dims = 2
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, dims=dims)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = conv_nd(dims, dim, hidden_dim * 3, 1, bias = False)
        self.to_out = conv_nd(dims, hidden_dim, dim, 1, bias = False)

        self.dims = dims

    def forward(self, x):
        if self.dims == 2:
            b, c, h, w = x.shape
        elif self.dims == 3:
            b, c, h, w, d = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        if self.dims == 2:
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
        elif self.dims == 3:
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h (x y z) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        if self.dims == 2:
            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        elif self.dims == 3:
            out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x=h, y=w, z=d)
            
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dims=2,
        dim=128,
        init_dim=None,
        out_dim=None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels=3,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,    # defaults to full attention only for inner most layer
        flash_attn=False
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = conv_nd(dims, channels, init_dim, 7, padding=3)

        dims_list = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims_list[:-1], dims_list[1:]))

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks
        FullAttention = partial(Attention, flash=flash_attn, dims=dims)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout, dims=dims)

        # layers
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads, dims=dims),
                Downsample(dim_in, dim_out, dims=dims) if not is_last else conv_nd(dims, dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims_list[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads, dims=dims),
                Upsample(dim_out, dim_in, dims=dims) if not is_last else conv_nd(dims, dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = conv_nd(dims, init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, times):

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(times)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# trainer

def cycle(dl):
    while True:
        for batch in dl:
            if isinstance(batch, tuple) and len(batch) == 2:
                source_batch, target_batch = batch
                if source_batch is not None:
                    yield source_batch, target_batch
                else:
                    yield target_batch
            else:
                yield batch

class Trainer(Module):
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        is_single_gpu = torch.cuda.device_count() == 1

        def remove_module_prefix(state_dict):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
    
        model_state = remove_module_prefix(checkpoint['model']) if is_single_gpu else checkpoint['model']
        self.model.load_state_dict(model_state)

        if 'ema_model' in checkpoint and self.ema_model is not None:
            ema_state = remove_module_prefix(checkpoint['ema_model']) if is_single_gpu else checkpoint['ema_model']
            self.ema_model.load_state_dict(ema_state)

        optimizer_state = remove_module_prefix(checkpoint['optimizer']) if is_single_gpu else checkpoint['optimizer']
        self.optimizer.load_state_dict(optimizer_state)

    def __init__(
        self,
        rectified_flow: dict | RectifiedFlow,
        *,
        train_dataset: Dataset,
        val_dataset: dict | None = None,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        results_folder: str = './experiments/rectified_flow',
        checkpoint_every: int = 1000,
        validation_every: int = 1000,
        log_loss_every: int = 10,  # 추가: loss 출력 주기
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True,
        resume_checkpoint: str = None,  # 추가: 이어서 학습할 체크포인트 경로
        start_step=0
    ):
        super().__init__()
        # self.accelerator = Accelerator(**accelerate_kwargs)
        self.accelerator = Accelerator(mixed_precision="fp16", **accelerate_kwargs)
        self.scaler = GradScaler()  # mixed precision을 위한 GradScaler 추가

        if isinstance(rectified_flow, dict):
            rectified_flow = RectifiedFlow(**rectified_flow)

        self.model = rectified_flow

        # determine whether to keep track of EMA (if not using consistency FM)
        # which will determine which model to use for sampling

        use_ema &= not self.model.use_consistency

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        # optimizer, dataloader, and all that

        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)

        self.start_step = 0
        if resume_checkpoint is not None:
            self.start_step = start_step
            self.load_checkpoint(resume_checkpoint)
        
        self.dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        if val_dataset is not None:
            self.val_dl = DataLoader(val_dataset, batch_size = 1, shuffle = False, drop_last = False)
        else:
            self.val_dl = None

        # self.model = torch.compile(self.model)
        # if self.ema_model is not None:
        #     self.ema_model = torch.compile(self.ema_model)
        
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps
        
        # folders

        self.checkpoints_folder = Path(os.path.join(results_folder, 'checkpoints'))
        self.img_folder = Path(os.path.join(results_folder, 'results'))
        self.log_folder = Path(os.path.join(results_folder, 'logs'))

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.img_folder.mkdir(exist_ok = True, parents = True)
        self.log_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.validation_every = validation_every
        self.log_loss_every = log_loss_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.img_folder.is_dir()
        assert self.log_folder.is_dir()

        # logging 설정
        # 현재 날짜와 시간을 기반으로 로그 파일명 생성
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(results_folder, f'{current_time}_train_log.txt')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # TensorBoard SummaryWriter 설정
        self.writer = SummaryWriter(log_dir=self.log_folder)

        # 파라미터를 로그에 기록
        self.log_parameters(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_samples=num_samples,
            use_ema=use_ema,
        )

    def log_parameters(self, **params):
        self.logger.info("Trainer initialized with the following parameters:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def forward(self):

        dl = cycle(self.dl)

        for step in range(self.start_step, self.num_train_steps):

            self.model.train()

            data = next(dl)
            if isinstance(data, list) and len(data) == 2:
                data_init, data, data_pos = data[0], data[1], None
            elif isinstance(data, list) and len(data) == 3:
                data_init, data, data_pos = data[0], data[1], data[2]
            else:
                data_init, data, data_pos = None, data, None

            if data_pos is not None:
                data_init = torch.cat([data_init, data_pos], dim=1)
            
            with autocast(device_type='cuda'):  # Mixed Precision 적용
                loss, loss_breakdown = self.model(data=data, data_init=data_init, return_loss_breakdown=True)

            self.log(loss_breakdown._asdict(), step = step)

            # TensorBoard에 loss 기록
            self.writer.add_scalar('Loss/Total', loss.item(), step)

            # loss 출력 및 로그 파일에 기록
            if divisible_by(step, self.log_loss_every):
                log_message = f'[{step}] loss: {loss.item():.3e}'
                self.logger.info(log_message)
    
            self.accelerator.backward(self.scaler.scale(loss))  # Mixed Precision
            self.scaler.step(self.optimizer)  # Optimizer step
            self.scaler.update()  # Scaler update

            # self.accelerator.backward(loss)
            # self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            if self.model.use_consistency:
                self.model.ema_model.update()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                eval_model = default(self.ema_model, self.model)

                if divisible_by(step, self.validation_every):
                    if self.val_dl is not None:
                        total_psnr = 0
                        total_ssim = 0
                        num_batches = 0

                        with torch.no_grad():
                            for idx, temp in enumerate(self.val_dl):
                                if len(temp) == 2:
                                    data_init, data = temp
                                elif len(temp) == 3:
                                    data_init, data, data_pos = temp
                                # Mixed Precision - Autocast 적용
                                with autocast(device_type='cuda'):
                                    sampled_trajectory = eval_model.sample(
                                        batch_size=min(self.num_samples, data_init.shape[0]),
                                        data_init=torch.cat([data_init, data_pos], dim=1) if data_pos is not None else data_init,
                                        data_shape=data.shape[1:]
                                    )

                                if len(sampled_trajectory.shape) == 5: # 2d
                                    save_image(sampled_trajectory[:,0,...], str(self.img_folder / f'{idx}_trajectory_{step}.png'), nrow=16)
                                else: # 3d
                                    save_image(sampled_trajectory[:,0,:,:,data_init.shape[-1]//2], str(self.img_folder / f'{idx}_trajectory_{step}.png'), nrow=16)

                                sampled = sampled_trajectory[-1]
                                metrics = self.compute_metrics(sampled, data)
                                print(idx, metrics)
                                total_psnr += metrics["psnr"]
                                total_ssim += metrics["ssim"]
                                num_batches += 1

                                for name, tensor in zip(['data_init', 'data', f'results_{step}'], [data_init, data, sampled]):
                                    if len(tensor.shape) == 4:
                                        save_image(tensor, str(self.img_folder / f'{idx}_{name}.png'))
                                    elif len(tensor.shape) == 5:
                                        save_image(rearrange(tensor[0].permute(3, 0, 1, 2), '(row col) c h w -> c (row h) (col w)', row=self.num_sample_rows), str(self.img_folder / f'{idx}_{name}.png'))
                                        # save_nifti(tensor, str(self.img_folder / f'{idx}_{name}.nii.gz'))

                        avg_psnr = total_psnr / num_batches
                        avg_ssim = total_ssim / num_batches

                        self.logger.info(f'Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
                        self.writer.add_scalar('Validation/PSNR', avg_psnr, step)
                        self.writer.add_scalar('Validation/SSIM', avg_ssim, step)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint_{step}.pt')

            self.accelerator.wait_for_everyone()
        
        # Training 완료 시 TensorBoard SummaryWriter 닫기
        self.writer.close()
        print('training complete')
    
    def compute_metrics(self, outputs, targets):
        outputs = outputs.cpu()
        targets = targets.cpu()

        psnr_value = psnr(targets, outputs)

        if len(outputs.shape) == 4:
            ssim_value = ssim(targets, outputs)
        elif len(outputs.shape) == 5:
            ssim_value = ssim3D(targets, outputs)

        return {
            "psnr": psnr_value,
            "ssim": ssim_value
        }
