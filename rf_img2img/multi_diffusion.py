import torch
from torch.nn import Module
from typing import Tuple, Literal
from torch.autograd import Variable
from einops import rearrange, repeat
from torchdiffeq import odeint
from tqdm import tqdm

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


class MultiDiffusion(Module):
    def __init__(self, 
                model: Module,
                device,
                time_cond_kwarg: str | None = 'times',
                odeint_kwargs: dict = dict(
                    atol = 1e-5,
                    rtol = 1e-5,
                    method = 'midpoint'
                    ),
                predict: Literal['flow', 'noise'] = 'flow',
                use_consistency = False,
                clip_during_sampling = False,
                clip_values: Tuple[float, float] = (-1., 1.),
                clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
                clip_flow_values: Tuple[float, float] = (-3., 3)
                 ) -> None:
        super().__init__()

        self.model = model
        self.device = device

        self.time_cond_kwarg = time_cond_kwarg
        self.odeint_kwargs = odeint_kwargs

        self.predict = predict
        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')
        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency flow matching

        self.use_consistency = use_consistency

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
            output = model.model(torch.cat([noised, data_init.to(noised.device)], dim=1), **model_kwargs)
        else:
            output = model.model(noised, **model_kwargs)

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

    # @torch.no_grad()
    # def double_sample(
    #     self,
    #     batches,
    #     ijk_patch_indices,
    #     batch_size = 1,
    #     steps = 16,
    #     data_shape: Tuple[int, ...] | None = None,
    #     use_ema: bool = False,
    #     **model_kwargs
    # ):
    #     use_ema = default(use_ema, self.use_consistency)
    #     assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

    #     self.eval()

    #     maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
    #     maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

    #     # ode step function

    #     def ode_fn(t, x):
    #         x = maybe_clip(x)

    #         _, flow = self.predict_flow(self.model, x, times = t, data_init=batch, **model_kwargs)

    #         flow = maybe_clip_flow(flow)

    #         return flow
        
    #     total_times = torch.linspace(0., 1., steps, device = self.device)
    #     times1 = total_times[:len(total_times)//2+1]
    #     times2 = total_times[len(total_times)//2:]

    #     # start with random gaussian noise - y0
    #     noise = torch.randn((batch_size, *data_shape), device = self.device)

    #     for times in [times1, times2]:
    #         value = torch.zeros((batch_size, *data_shape), device = self.device)
    #         weight = torch.zeros_like(value, device = self.device)

    #         for i in tqdm(range(len(batches))):
    #             batch = batches[i]
    #             batch = Variable(batch.to(self.device))

    #             # Retrieve the patch indices
    #             if len(ijk_patch_indices[i]) == 4:
    #                 istart, iend, jstart, jend = ijk_patch_indices[i]
    #                 noise_patch = noise[:, :, istart:iend, jstart:jend]
    #             elif len(ijk_patch_indices[i]) == 6:
    #                 istart, iend, jstart, jend, kstart, kend = ijk_patch_indices[i]
    #                 noise_patch = noise[:, :, istart:iend, jstart:jend, kstart:kend]

                
    #             # Solve the ODE for the current patch and time range
    #             pred_patch = odeint(ode_fn, noise_patch, times, **self.odeint_kwargs)[-1]

    #             # Accumulate the predictions and update the weight
    #             if len(ijk_patch_indices[i]) == 4:
    #                 value[:, :, istart:iend, jstart:jend] += pred_patch
    #                 weight[:, :, istart:iend, jstart:jend] += 1.0
    #             elif len(ijk_patch_indices[i]) == 6:
    #                 value[:, :, istart:iend, jstart:jend, kstart:kend] += pred_patch
    #                 weight[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

    #         # Update value and weight after each time range
    #         noise = torch.where(weight > 0, value / weight, value)
            
    #     sampled_data = noise

    #     print("Evaluation complete")

    #     return sampled_data
    
    @torch.no_grad()
    def simple_sample(
        self,
        ijk_patch_indices,
        batch_size = 1,
        steps = 16,
        data_init = None,
        data_shape: Tuple[int, ...] | None = None,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        self.eval()

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity
        
        times = torch.linspace(0., 1., steps, device = self.device)

        # start with random gaussian noise - y0
        noise = torch.randn((batch_size, *data_shape), device = self.device)

        value = torch.zeros((batch_size, *data_shape), device=self.device)
        weight = torch.zeros_like(value, device=self.device)

        for i in tqdm(range(len(ijk_patch_indices))):
            if len(ijk_patch_indices[i]) == 4:
                istart, iend, jstart, jend = ijk_patch_indices[i]
                noise_patch = noise[:, :, istart:iend, jstart:jend]
                data_patch = data_init[:, :, istart:iend, jstart:jend]
            elif len(ijk_patch_indices[i]) == 6:
                istart, iend, jstart, jend, kstart, kend = ijk_patch_indices[i]
                noise_patch = noise[:, :, istart:iend, jstart:jend, kstart:kend]
                data_patch = data_init[:, :, istart:iend, jstart:jend, kstart:kend]

            # Solve the ODE using the odeint solver
            def ode_fn(t, x):
                x = maybe_clip(x)

                _, flow = self.predict_flow(self.model, x, times=t, data_init=data_patch, **model_kwargs)

                flow = maybe_clip_flow(flow)

                return flow
            
            pred_patch = odeint(ode_fn, noise_patch, times, **self.odeint_kwargs)[-1]

            # Accumulate the predictions and update the weight
            if len(ijk_patch_indices[i]) == 4:
                value[:, :, istart:iend, jstart:jend] += pred_patch
                weight[:, :, istart:iend, jstart:jend] += 1.0
            elif len(ijk_patch_indices[i]) == 6:
                value[:, :, istart:iend, jstart:jend, kstart:kend] += pred_patch
                weight[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

        sampled_data = torch.where(weight > 0, value / weight, value)

        return sampled_data
    
    @torch.no_grad()
    def multi_sample(
        self,
        ijk_patch_indices,
        batch_size = 1,
        steps = 16,
        data_init = None,
        data_shape: Tuple[int, ...] | None = None,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        self.eval()

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity
        
        times = torch.linspace(0., 1., steps, device = self.device)
        
        # Compute the time step size (delta_t)
        delta_t = times[1] - times[0]

        # start with random gaussian noise - y0
        noise = torch.randn((batch_size, *data_shape), device = self.device)

        trajectory = []
        trajectory.append(noise)

        for step in tqdm(range(steps-1)):
            value = torch.zeros((batch_size, *data_shape), device = self.device)
            weight = torch.zeros_like(value, device = self.device)

            for i in range(len(ijk_patch_indices)): 
                if len(ijk_patch_indices[i]) == 4:
                    istart, iend, jstart, jend = ijk_patch_indices[i]
                    noise_patch = noise[:, :, istart:iend, jstart:jend]
                    data_patch = data_init[:, :, istart:iend, jstart:jend]
                elif len(ijk_patch_indices[i]) == 6:
                    istart, iend, jstart, jend, kstart, kend = ijk_patch_indices[i]
                    noise_patch = noise[:, :, istart:iend, jstart:jend, kstart:kend]
                    data_patch = data_init[:, :, istart:iend, jstart:jend, kstart:kend]

                # Directly compute the flow for this step without odeint
                def ode_fn(t, x):
                    x = maybe_clip(x)

                    _, flow = self.predict_flow(self.model, x, times=t, data_init=data_patch, **model_kwargs)

                    flow = maybe_clip_flow(flow)

                    return flow
                
                pred_patch = odeint(ode_fn, noise_patch, times[step:step+2], **self.odeint_kwargs)[-1]
                #pred_patch = noise_patch + delta_t * ode_fn(times[step] + delta_t / 2, noise_patch + delta_t / 2 * ode_fn(times[step], noise_patch))

                # Accumulate the predictions and update the weight
                if len(ijk_patch_indices[i]) == 4:
                    value[:, :, istart:iend, jstart:jend] += pred_patch
                    weight[:, :, istart:iend, jstart:jend] += 1.0
                elif len(ijk_patch_indices[i]) == 6:
                    value[:, :, istart:iend, jstart:jend, kstart:kend] += pred_patch
                    weight[:, :, istart:iend, jstart:jend, kstart:kend] += 1.0

            # Update the noise to be the average of accumulated results so far
            noise = torch.where(weight > 0, value / weight, value)

            trajectory.append(noise)

        trajectory = torch.stack(trajectory, dim=0)
        
        #sampled_data = noise

        return trajectory #sampled_data