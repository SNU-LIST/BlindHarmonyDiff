import argparse
import torch
from rf_img2img import RectifiedFlow, Unet
from torch.utils.data import DataLoader, DistributedSampler
from dataset.harmony_dataset import HarmonyDataset, crop
import nibabel as nib
from pathlib import Path
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
import os
import numpy as np
from rf_img2img.metrics import psnr, ssim, ssim3D
import time
from einops import rearrange
from torchvision.utils import save_image

from torch.utils.data import Sampler
import math

def get_args():
    parser = argparse.ArgumentParser(description="Distributed Testing Script for Rectified Flow")

    # General settings
    parser.add_argument("--dims", type=int, default=3, help="Set dimensions for the model (2D or 3D)")
    parser.add_argument("--in-channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--patch-size", type=int, default=1, help="Patch size for the input data")
    parser.add_argument("--image-size", type=tuple, default=(144, 208, 92), help="Size of the input image")
    
    # Paths
    parser.add_argument("--checkpoint-path", type=str, default='/home/hwihun/blindharmony_diff/RF_winsor/checkpoints/3d_n_patch_02_150000.pt', help="Path to the model checkpoint")

    parser.add_argument("--edge-path", type=str, default='/fast_storage/hwihun/pkl_blindharmony/pkl_T2_registered_edge0.08_T2_test.pklv4', help="Path to the edge dataset")
    parser.add_argument("--image-path", type=str, default='/fast_storage/hwihun/pkl_blindharmony/pkl_T2_registered_T2_test.pklv4', help="Path to the image dataset")

    parser.add_argument("--name", type=str, default='test_log', help="Directory to save results")
    parser.add_argument("--subject-id", type=int, default=None, help="Validation only for specific subject")

    # GPU settings
    parser.add_argument("--master-addr", type=str, default='127.0.0.1', help="Master address for distributed training")
    parser.add_argument("--master-port", type=str, default='29700', help="Master port for distributed training")
    parser.add_argument("--cuda-visible-devices", type=str, default="2", help="Visible CUDA devices")

    return parser.parse_args()

def save_nifti(volume, output_path, voxel_size=(1.2, 1.2, 1.25), origin=(0, 0, 0)):
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]  # X축 해상도
    affine[1, 1] = voxel_size[1]  # Y축 해상도
    affine[2, 2] = voxel_size[2]  # Z축 해상도
    affine[:3, 3] = origin  # 좌표계 원점 설정

    nifti_img = nib.Nifti1Image(volume[0,:,:,:].cpu().numpy(), affine=affine)
    nib.save(nifti_img, output_path)

def compute_metrics(outputs, targets, is_mask=False):
    outputs = outputs.cpu()
    targets = targets.cpu()

    psnr_value = psnr(targets, outputs, is_mask=is_mask)

    if len(outputs.shape) == 4:
        ssim_value = ssim(targets, outputs, is_mask=is_mask)
    elif len(outputs.shape) == 5:
        ssim_value = ssim3D(targets, outputs, is_mask=is_mask)

    return {
        "psnr": psnr_value,
        "ssim": ssim_value
    }

def load_model_from_checkpoint(checkpoint_path, device, dims, in_channels, out_channels):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = Unet(dims=dims, channels=in_channels, out_dim=out_channels, dim=64)
    rectified_flow = RectifiedFlow(model=model)

    rectified_flow.load_state_dict(new_state_dict, strict=True)
    rectified_flow.to(device)

    # 모델을 컴파일하여 최적화 (torch.compile() 추가)
    rectified_flow = torch.compile(rectified_flow)

    rectified_flow.eval()
    return rectified_flow

class BatchDistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()
        self.shuffle = shuffle

        # 배치 단위로 나눌 데이터셋 인덱스 계산
        self.num_samples = int(math.ceil(len(self.dataset) / self.batch_size))
        self.total_size = self.num_samples * self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # shuffle 옵션 적용
        if self.shuffle:
            np.random.shuffle(indices)

        # 전체 배치를 나누고, 각 rank에 해당하는 배치 인덱스를 할당
        batch_indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        batch_indices = batch_indices[self.rank::self.num_replicas]

        # flatten하여 반환 (DataLoader에서 사용 가능하도록)
        batch_indices = [item for sublist in batch_indices for item in sublist]
        return iter(batch_indices)

    def __len__(self):
        return self.num_samples // self.num_replicas
    
def load_dataset(edge_path, image_path, micro_batch_size, world_size, rank, dims, patch_size, subject_id):
    test_dataset = HarmonyDataset(
        dims=dims,
        edge_path=edge_path,
        image_path=image_path,
        # image_size=image_size[:dims],
        subject_id=subject_id,
        patch_size=patch_size,
        is_train=False
    )

    if world_size > 1:
        # sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        sampler = BatchDistributedSampler(
            test_dataset, batch_size=micro_batch_size, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_loader = DataLoader(test_dataset, batch_size=micro_batch_size, shuffle=False, sampler=sampler)
    else:
        test_loader = DataLoader(test_dataset, batch_size=micro_batch_size, shuffle=False)
    return test_loader

def gather_samples(rank, world_size, tensor):
    if world_size > 1:
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        ordered_tensors = [gathered_tensors[i] for i in range(world_size)]
        return torch.cat(ordered_tensors, dim=0)
        # return torch.cat(gathered_tensors, dim=0)
    return tensor

def log_metrics(idx, metrics, time_taken, log_file):
    with open(log_file, 'a') as f:
        f.write(f"Idx: {idx}, PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}, Time: {time_taken:.4f} seconds\n")

def test_model(rank, world_size, args):
    
    # GPU 1개인 경우 dist.init_process_group 생략
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # 모델 로드
    model = load_model_from_checkpoint(args.checkpoint_path, device, args.dims, args.in_channels, args.out_channels)

    # DDP 적용 (world_size가 1이면 DDP 사용하지 않음)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    micro_batch_size = args.image_size[-1] // world_size if args.dims == 2 else torch.cuda.device_count()
    test_loader = load_dataset(args.edge_path, args.image_path, micro_batch_size, world_size, rank, args.dims, args.patch_size, args.subject_id)


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_file = f'{args.output_dir}/{args.name}.txt'

    total_psnr, total_ssim, num_batches = 0, 0, 0

    with torch.no_grad():
        for idx, temp in enumerate(test_loader):
            if len(temp) == 2:
                data_init, data = temp
                data_pos = None
            elif len(temp) == 3:
                data_init, data, data_pos = temp

            data_init = data_init.to(device)
            data = data.to(device)
            if data_pos is not None:
                data_pos = data_pos.to(device)

            start_time = time.time()

            if world_size > 1:
                sampled = model.module.sample(
                    batch_size=data_init.shape[0],
                    data_init=torch.cat([data_init, data_pos], dim=1) if data_pos is not None else data_init,
                    data_shape=data_init.shape[1:]
                )[-1]
            else:
                sampled = model.sample(
                    batch_size=data_init.shape[0],
                    data_init=torch.cat([data_init, data_pos], dim=1) if data_pos is not None else data_init,
                    data_shape=data_init.shape[1:]
                )[-1]

            end_time = time.time()
            time_taken = end_time - start_time

            if args.dims == 2 and world_size > 1:
                sampled = gather_samples(rank, world_size, sampled).permute(1, 2, 3, 0)
                data = gather_samples(rank, world_size, data).permute(1, 2, 3, 0)
                data_init = gather_samples(rank, world_size, data_init).permute(1, 2, 3, 0)

            

            # 크롭 적용 (원본 크기로 복원)
            if args.dims ==3 and sampled.shape[-3:] != args.image_size:
                sampled = crop(sampled[0], args.image_size)
                data = crop(data[0], args.image_size)
                data_init = crop(data_init[0], args.image_size)

            if rank == 0:
                metrics = compute_metrics(sampled, data)
                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]
                print(metrics)
                num_batches += 1
                log_metrics(idx, metrics, time_taken, log_file)

                for name, tensor in zip(['label', 'output'], [data, sampled]): #zip(['data_init', 'data', 'output'], [data_init, data, sampled]):
                    if args.dims == 2:
                        save_nifti(tensor.permute(1,2,3,0), f'{args.output_dir}/{idx}_{name}.nii.gz')
                    else:
                        save_nifti(tensor, f'{args.output_dir}/{idx}_{name}.nii.gz')
                    # tensor = rearrange(tensor.permute(3, 0, 1, 2), '(row col) c h w -> c (row h) (col w)', row=4)
                    # save_image(tensor, f'{args.output_dir}/{idx}_{name}.png')

        if rank == 0:
            avg_psnr = total_psnr / num_batches
            avg_ssim = total_ssim / num_batches
            print(f'Averaged Test - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

            with open(log_file, 'a') as f:
                f.write(f'Averaged Test - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    tmp_name = args.image_path.split('/')[-1][:-6]
    args.output_dir = f'./results/{args.dims}d/{tmp_name}'
    
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda_visible_devices

    world_size = torch.cuda.device_count()

    out_channels = args.in_channels
    in_channels = args.in_channels * 2
    if args.patch_size is not None:
        in_channels += args.dims
    args.in_channels = in_channels
    args.out_channels = out_channels

    if world_size > 1:
        spawn(test_model, args=(world_size, args), nprocs=world_size, join=True)
    else:
        test_model(0, world_size, args)  # world_size가 1일 경우 rank=0으로 실행
