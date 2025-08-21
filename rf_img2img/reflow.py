from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import Module

from rf_img2img.rectified_flow import RectifiedFlow

from ema_pytorch import EMA
from accelerate import Accelerator

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import logging
import os
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import math
from torchvision.utils import save_image
from einops import rearrange
from typing import Tuple

from .rectified_flow import save_nifti
from .metrics import psnr, ssim, ssim3D

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(dl):
    while True:
        for batch in dl:
            # batch가 튜플인 경우 (source_batch, target_batch)
            if isinstance(batch, tuple) and len(batch) == 2:
                source_batch, target_batch = batch
                if source_batch is not None:
                    yield source_batch, target_batch
                else:
                    yield target_batch
            # batch가 단일 요소 (target_batch)만 있는 경우
            else:
                yield batch
    
def divisible_by(num, den):
    return (num % den) == 0

# reflow wrapper

class Reflow(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        frozen_model: RectifiedFlow | None = None,
        *,
        batch_size = 16,
        data_shape: Tuple[int, ...] | None = None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.data_shape = data_shape

        self.model = rectified_flow

        if not exists(frozen_model):
            # make a frozen copy of the model and set requires grad to be False for all parameters for safe measure

            frozen_model = deepcopy(rectified_flow)

            for p in frozen_model.parameters():
                p.detach_()

        self.frozen_model = frozen_model

    def device(self):
        return next(self.parameters()).device

    def parameters(self):
        return self.model.parameters() # omit frozen model

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def forward(self, data, data_init, batch_size: int = 16):
        _, *data_shape = data.shape
        self.data_shape = default(self.data_shape, data_shape)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # hy

        sampled_output = self.frozen_model.sample( # rectified flow
                    batch_size=batch_size,
                    data_init=data_init,
                    data_shape=data.shape[1:])

        # the coupling in the paper is (noise, sampled_output)
        loss = self.model(data = sampled_output, data_init = data_init)

        return loss

# reflow trainer

class ReflowTrainer(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        *,
        train_dataset: dict | Dataset, # hy
        val_dataset: dict | Dataset, #hy
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        results_folder: str = './experiments/reflow',
        validation_every: int = 100,
        checkpoint_every: int = 1000,
        log_loss_every: int = 10,  # 추가: loss 출력 주기
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        #assert not rectified_flow.use_consistency, 'reflow is not needed if using consistency flow matching'

        self.model = Reflow(rectified_flow)

        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
        self.dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        if val_dataset is not None:
            self.val_dl = DataLoader(val_dataset, batch_size=num_samples, shuffle=False, drop_last=False)
        else:
            self.val_dl = None
        
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.batch_size = batch_size
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
            num_samples=num_samples
        )

    def log_parameters(self, **params):
        self.logger.info("Trainer initialized with the following parameters:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def forward(self):

        dl = cycle(self.dl) # hy

        for step in range(self.num_train_steps):

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

            loss = self.model(data=data, data_init=data_init, batch_size=self.batch_size) # Reflow

            self.log(loss, step = step)

            # TensorBoard에 loss 기록
            self.writer.add_scalar('Loss/Total', loss.item(), step)

            # loss 출력 및 로그 파일에 기록
            if divisible_by(step, self.log_loss_every):
                log_message = f'[{step}] reflow loss: {loss.item():.3e}'
                self.logger.info(log_message)

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if self.val_dl is not None and divisible_by(step, self.validation_every):
                    total_psnr = 0
                    total_ssim = 0
                    num_batches = 0

                    with torch.no_grad():
                        for idx, temp in enumerate(self.val_dl):
                            if len(temp) == 2:
                                data_init, data = temp
                            elif len(temp) == 3:
                                data_init, data, data_pos = temp

                            sampled = self.ema_model.sample(
                                batch_size=min(self.num_samples, data_init.shape[0]),
                                data_init=torch.cat([data_init, data_pos], dim=1) if data_pos is not None else data_init,
                                data_shape=data.shape[1:]
                            )

                            metrics = self.compute_metrics(sampled, data)
                            print(idx, metrics)
                            total_psnr += metrics["psnr"]
                            total_ssim += metrics["ssim"]
                            num_batches += 1

                            for name, tensor in zip(['data_init', 'data', f'results_{step}'], [data_init, data, sampled]):
                                tensor = (tensor + 1) / 2
                                if len(tensor.shape) == 4:
                                    tensor.clamp_(0., 1.)
                                    tensor = rearrange(tensor, '(row col) c h w -> c (row h) (col w)', row=self.num_sample_rows)
                                    save_image(tensor, str(self.img_folder / f'{idx}_{name}.png'))
                                elif len(tensor.shape) == 5:
                                    save_image(rearrange(tensor[0].permute(3, 0, 1, 2), '(row col) c h w -> c (row h) (col w)', row=self.num_sample_rows), str(self.img_folder / f'{idx}_{name}.png'))
                                    save_nifti(tensor, str(self.img_folder / f'{idx}_{name}.nii.gz'))

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
        print('reflow training complete')

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
