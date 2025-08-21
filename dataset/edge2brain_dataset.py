import h5py
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import random
import numpy as np


def patchify2D(image, patch_size, padding=None, coordinate=None):
    device = image.device
    _, h_resolution, w_resolution = image.shape  # Batch 차원이 제거된 상태이므로 (C, H, W)

    if padding is not None:
        padded = torch.zeros((image.size(0), image.size(1) + padding * 2,
                              image.size(2) + padding * 2), dtype=image.dtype, device=device)
        padded[:, padding:-padding, padding:-padding] = image
    else:
        padded = image

    h, w = padded.size(1), padded.size(2)  # H, W 크기
    th, tw = patch_size, patch_size
    if coordinate is None:
        if w == tw and h == th:
            i = torch.zeros(1, device=device).long()
            j = torch.zeros(1, device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (1,), device=device)
            j = torch.randint(0, w - tw + 1, (1,), device=device)
    else:
        i, j = coordinate

    rows = torch.arange(th, dtype=torch.long, device=device) + i
    columns = torch.arange(tw, dtype=torch.long, device=device) + j
    padded = padded[:, rows[:, None], columns[None, :]]

    x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0)
    y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0)
    x_pos = x_pos + j
    y_pos = y_pos + i
    x_pos = (x_pos / (w_resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (h_resolution - 1) - 0.5) * 2.
    image_pos = torch.cat((x_pos, y_pos), dim=0)

    return padded, image_pos, (i, j)

def patchify3D(image, patch_size, padding=None, coordinate=None):
    device = image.device
    c, h_resolution, w_resolution, d_resolution = image.shape  # Remove batch dimension

    if padding is not None:
        padded = torch.zeros((image.size(0), image.size(1) + padding * 2,
                              image.size(2) + padding * 2, image.size(3) + padding * 2),
                             dtype=image.dtype, device=device)
        padded[:, padding:-padding, padding:-padding, padding:-padding] = image
    else:
        padded = image

    h, w, d = padded.size(1), padded.size(2), padded.size(3)
    th, tw, td = patch_size, patch_size, patch_size
    if coordinate is None:
        if w == tw and h == th and d == td:
            i = torch.zeros((1,), device=device).long()  # Single index without batch dimension
            j = torch.zeros((1,), device=device).long()
            k = torch.zeros((1,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (1,), device=device)
            j = torch.randint(0, w - tw + 1, (1,), device=device)
            k = torch.randint(0, d - td + 1, (1,), device=device)
    else:
        i, j, k = coordinate

    rows = torch.arange(th, dtype=torch.long, device=device) + i
    columns = torch.arange(tw, dtype=torch.long, device=device) + j
    depths = torch.arange(td, dtype=torch.long, device=device) + k

    # Extract patches
    padded = padded.permute(0, 1, 2, 3)
    padded = padded[:, rows[:, None, None], columns[None, :, None], depths[None, None, :]]
    
    # X, Y, Z positional calculations
    x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(2).repeat(1, 1, td).unsqueeze(0).repeat(1, 1, 1, 1)
    y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(2).repeat(1, 1, td).unsqueeze(0).repeat(1, 1, 1, 1)
    z_pos = torch.arange(td, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0).repeat(th, tw, 1).unsqueeze(0).repeat(1, 1, 1, 1)

    # Corrected to use the correct indices
    x_pos = x_pos + j.view(-1, 1, 1)
    y_pos = y_pos + i.view(-1, 1, 1)
    z_pos = z_pos + k.view(-1, 1, 1)

    # Normalization to range [-1, 1]
    x_pos = (x_pos / (w_resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (h_resolution - 1) - 0.5) * 2.
    z_pos = (z_pos / (d_resolution - 1) - 0.5) * 2.

    images_pos = torch.cat((x_pos, y_pos, z_pos), dim=0)

    return padded, images_pos, (i, j, k)


# 2d

def build_train_transform(height, width):
    return A.Compose([
        A.CenterCrop(height=height, width=width),
        A.HorizontalFlip(p=0.5),  # Apply the same flip to both image and mask
        ToTensorV2()
    ])

def build_test_transform(height, width):
    return A.Compose([
        A.CenterCrop(height=height, width=width),  # Center crop to 144x192
        ToTensorV2()
    ])

class Edge2Brain2DDataset(Dataset):
    def __init__(self, path, patch_size=None, subject_id=None, train=True, transform=None, image_size=None):
        """
        Args:
            path (str): Path to the HDF5 file containing both image and edge datasets.
            subject_id (int, optional): Use only this specific subject ID (e.g., 1 for subject1).
            train (bool): Whether the dataset is used for training or testing.
            transform (callable, optional): Optional transform to be applied on both images.
        """
        self.path = path
        self.train = train
        self.patch_size = patch_size
        self.patchify = patchify2D

        # Open the HDF5 file and load metadata (number of subjects and slices)
        with h5py.File(self.path, 'r') as h5_file:
            self.num_slices = h5_file['image'].shape[3]  # Assuming shape is (num_subjects, height, width, slices)
            self.num_subjects = h5_file['image'].shape[0]  # The number of subjects (3D images)

        # 특정 서브젝트만 사용할 경우 해당 서브젝트에 대한 슬라이스만 로드
        if subject_id is not None:
            assert subject_id < self.num_subjects, f"Invalid subject_id {subject_id}. Max is {self.num_subjects - 1}."
            self.subject_id = subject_id
            self.total_images = self.num_slices  # 특정 서브젝트의 슬라이스 수만 사용
        else:
            self.subject_id = None
            self.total_images = self.num_subjects * self.num_slices  # 모든 서브젝트의 슬라이스 수

        # Set up transformations
        if isinstance(image_size, int):
            h, w = image_size, image_size
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            h, w = image_size
        if transform is None:
            self.transform = build_train_transform(h, w) if train else build_test_transform(h, w)
        else:
            self.transform = transform

    def __len__(self):
        return self.total_images

    def load_single_slice(self, subject_idx, slice_idx, data_type='image'):
        with h5py.File(self.path, 'r') as h5_file:
            if data_type in h5_file:
                slice_data = h5_file[data_type][subject_idx, :, :, slice_idx]
                if data_type == 'image':
                    min_val = h5_file[data_type][subject_idx].min()
                    max_val = h5_file[data_type][subject_idx].max()
                    if min_val == max_val:
                        slice_data = slice_data - min_val
                    else:
                        slice_data = (slice_data - min_val) / (max_val - min_val)
                slice_data = slice_data * 2 - 1  # [-1, 1]로 변환
            else:
                raise ValueError(f"Dataset '{data_type}' not found in the file.")
        
        return slice_data  # Albumentations 변환을 위해 numpy 배열로 유지

    def __getitem__(self, idx):
        if self.subject_id is not None:
            # 주어진 subject의 슬라이스 인덱스를 바로 계산
            subject_idx = self.subject_id
            slice_idx = idx
        else:
            # 모든 서브젝트를 사용할 경우 인덱스에 맞게 서브젝트와 슬라이스 계산
            subject_idx = idx // self.num_slices  # 3D 이미지의 인덱스
            slice_idx = idx % self.num_slices  # 3D 이미지 내 슬라이스 인덱스

        # Lazy load the data for the specific slice from both the 'edge' and 'image' datasets
        edge_slice = self.load_single_slice(subject_idx, slice_idx, data_type='edge')
        image_slice = self.load_single_slice(subject_idx, slice_idx, data_type='image')

        # Apply transformations to both image and edge (as mask)
        if self.transform:
            transformed = self.transform(image=image_slice, mask=edge_slice)
            image_slice = transformed['image']
            edge_slice = transformed['mask']
            edge_slice = edge_slice.unsqueeze(0).float()  # (H, W) -> (1, H, W), float32로 변환
            image_slice = image_slice.float()  # image_slice도 float32로 변환
        
        if self.patch_size is None:
            return edge_slice, image_slice
        
        else:
            if self.train:
                edge_slice, data_pos, coordinate = self.patchify(edge_slice, self.patch_size)
                image_slice, _, _ = self.patchify(image_slice, self.patch_size, coordinate=coordinate)
            else:
                _, h, w = edge_slice.shape
                x_start, y_start = 0, 0
                x_pos = torch.arange(x_start, x_start+w).view(1, -1).repeat(h, 1)
                y_pos = torch.arange(y_start, y_start+h).view(-1, 1).repeat(1, w)
                x_pos = (x_pos / (w - 1) - 0.5) * 2.
                y_pos = (y_pos / (h - 1) - 0.5) * 2.
                data_pos = torch.stack([x_pos, y_pos], dim=0)

            return edge_slice, image_slice, data_pos
    


# 3d

class CenterCrop3D:
    def __init__(self, height, width, depth, train=False):
        self.height = height
        self.width = width
        self.depth = depth
        self.train = train  # Whether to apply random flips in training

    def __call__(self, image, edge):
        assert image.ndim == 4 and edge.ndim == 4, "Input should be 4D tensors with shape (C, H, W, D)"
        
        # Get original dimensions
        _, orig_height, orig_width, orig_depth = image.shape

        # Calculate starting indices for crop
        d_start = (orig_depth - self.depth) // 2 if orig_depth > self.depth else 0
        h_start = (orig_height - self.height) // 2 if orig_height > self.height else 0
        w_start = (orig_width - self.width) // 2 if orig_width > self.width else 0

        # Perform the crop on both image and edge
        cropped_image = image[
            :,  # Keep all channels
            h_start:h_start + self.height,
            w_start:w_start + self.width,
            d_start:d_start + self.depth
        ]

        cropped_edge = edge[
            :,  # Keep all channels
            h_start:h_start + self.height,
            w_start:w_start + self.width,
            d_start:d_start + self.depth
        ]

        # Apply random flips during training
        if self.train:
            # Horizontal flip with probability hflip_prob (flip along height axis 1)
            if random.random() < 0.5:
                cropped_image = torch.flip(cropped_image, dims=[1])  # Flip along height (H)
                cropped_edge = torch.flip(cropped_edge, dims=[1])  # Apply same flip to edge

            # Vertical flip with probability vflip_prob (flip along width axis 2)
            if random.random() < 0.5:
                cropped_image = torch.flip(cropped_image, dims=[2])  # Flip along width (W)
                cropped_edge = torch.flip(cropped_edge, dims=[2])  # Apply same flip to edge

        return cropped_image, cropped_edge


class Edge2Brain3DDataset(Dataset):
    def __init__(self, path, patch_size=None, subject_id=None, train=True, transform=None, image_size=None):

        self.path = path
        self.train = train
        self.patch_size = patch_size
        self.patchify = patchify3D

        # Open the HDF5 file and load metadata (number of subjects and 3D volume size)
        with h5py.File(self.path, 'r') as h5_file:
            self.num_subjects = h5_file['image'].shape[0]  # The number of subjects (3D images)

        # 특정 서브젝트만 사용할 경우
        self.subject_id = subject_id
        if subject_id is not None:
            assert subject_id < self.num_subjects, f"Invalid subject_id {subject_id}. Max is {self.num_subjects - 1}."
            self.subject_id = subject_id
            self.num_subjects = 1  # 특정 서브젝트

        # Set up transformations
        if isinstance(image_size, int):
            h, w, d = image_size, image_size, image_size
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 3:
            h, w, d = image_size
        if transform is None and image_size is not None:
            self.transform = CenterCrop3D(depth=d, height=h, width=w, train=train)

    def __len__(self):
        return self.num_subjects

    def load_full_volume(self, subject_idx, data_type='image'):
        with h5py.File(self.path, 'r') as h5_file:
            if data_type in h5_file:
                volume_data = h5_file[data_type][subject_idx]  # Full 3D volume
                if data_type == 'image':
                    min_val = volume_data.min()
                    max_val = volume_data.max()
                    if min_val == max_val:
                        volume_data = volume_data - min_val
                    else:
                        volume_data = (volume_data - min_val) / (max_val - min_val)
                volume_data = volume_data * 2 - 1  # [-1, 1]
        
        # Convert volume_data to [C, H, W, D] (add channel dimension C=1)
        #volume_data = np.expand_dims(volume_data, axis=0)  # (1, H, W, D)
        volume_data = torch.tensor(volume_data, dtype=torch.float32).unsqueeze(0).float() # (1, H, W, D)
        return volume_data

    def __getitem__(self, idx):
        if self.subject_id is not None:
            subject_idx = self.subject_id
        else:
            subject_idx = idx

        # Load the 3D volume data for both the 'edge' and 'image' datasets
        edge_volume = self.load_full_volume(subject_idx, data_type='edge')
        image_volume = self.load_full_volume(subject_idx, data_type='image')

        # Apply CenterCrop3D first (both image and edge together)
        if self.transform is not None:
            image_volume, edge_volume = self.transform(image_volume, edge_volume)
        
        if self.patch_size is None:
            return edge_volume, image_volume
        
        else:
            if self.train:
                edge_volume, data_pos, coordinate = self.patchify(edge_volume, self.patch_size)
                image_volume, _, _ = self.patchify(image_volume, self.patch_size, coordinate=coordinate)    
            else:
                _, h, w, d = edge_volume.shape
                x_start, y_start, z_start = 0, 0, 0
                x_pos = torch.arange(x_start, x_start+w).view(1, -1, 1).repeat(h, 1, d)
                y_pos = torch.arange(y_start, y_start+h).view(-1, 1, 1).repeat(1, w, d)
                z_pos = torch.arange(z_start, z_start+d).view(1, 1, -1).repeat(h, w, 1)
                x_pos = (x_pos / (w - 1) - 0.5) * 2.
                y_pos = (y_pos / (h - 1) - 0.5) * 2.
                z_pos = (z_pos / (d - 1) - 0.5) * 2.
                data_pos = torch.stack([x_pos, y_pos, z_pos], dim=0)

            return edge_volume, image_volume, data_pos


class Edge2BrainDataset(Dataset):
    def __init__(self, dims, *args, **kwargs):
        if dims == 2:
            self.dataset = Edge2Brain2DDataset(*args, **kwargs)
        elif dims == 3:
            self.dataset = Edge2Brain3DDataset(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported dims: {dims}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]