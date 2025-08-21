from torch.utils.data import Dataset
import torch
import random
import pickle
import torch.nn.functional as F

def multiresol_patchify2D(edge, image, patch_size, is_multiresol=True):
    assert edge.shape == image.shape

    device = edge.device
    _, h, w = edge.shape
    
    crop_size = []
    coordinate = []
    for ms in [h, w]:
        if not is_multiresol or torch.rand(1).item() > 0.8:
            cs = patch_size
        else:
            cs = torch.randint(patch_size, ms, (1,)).item()
        crop_size.append(cs)
        if ms == cs:
            cor = torch.zeros((1,), device=device)
        else:
            cor = torch.randint(0, ms - cs + 1, (1,), device=device)
        coordinate.append(cor)
        
    i, j = coordinate
    th, tw = crop_size

    rows = torch.arange(th, device=device) + i
    columns = torch.arange(tw, device=device) + j

    edge_patch = edge[:, rows[:, None], columns[None, :]]
    image_patch = image[:, rows[:, None], columns[None, :]]
    x_pos, y_pos = rows, columns

    if is_multiresol:
        edge_patch = F.interpolate(edge_patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=True).squeeze(0)
        edge_patch = torch.where(edge_patch >= 0.4, torch.tensor(1.0), torch.tensor(0.0))
        image_patch = F.interpolate(image_patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=True).squeeze(0)

        x_pos = F.interpolate(x_pos.unsqueeze(0).unsqueeze(0).float(), size=patch_size, mode='linear', align_corners=True).squeeze()
        y_pos = F.interpolate(y_pos.unsqueeze(0).unsqueeze(0).float(), size=patch_size, mode='linear', align_corners=True).squeeze()
    
    x_pos = x_pos.view(-1, 1).repeat(1, patch_size)
    y_pos = y_pos.view(1, -1).repeat(patch_size, 1)
    
    # Normalization to range [-1, 1]
    x_pos = (x_pos / (h - 1) - 0.5) * 2.
    y_pos = (y_pos / (w - 1) - 0.5) * 2.

    data_pos = torch.stack((x_pos, y_pos), dim=0)

    return edge_patch, image_patch, data_pos

def multiresol_patchify3D(edge, image, patch_size, is_multiresol=True):
    assert edge.shape == image.shape

    device = edge.device
    _, h, w, d = edge.shape
    
    crop_size = []
    coordinate = []
    for ms in [h, w, d]:
        if not is_multiresol or torch.rand(1).item() > 0.8:
            cs = patch_size
        else:
            cs = torch.randint(patch_size, ms, (1,)).item()
        crop_size.append(cs)
        if ms == cs:
            cor = torch.zeros((1,), device=device)
        else:
            cor = torch.randint(0, ms - cs + 1, (1,), device=device)
        coordinate.append(cor)
        
    i, j, k = coordinate
    th, tw, td = crop_size

    rows = torch.arange(th, device=device) + i
    columns = torch.arange(tw, device=device) + j
    depths = torch.arange(td, device=device) + k

    # Extract patches
    edge_patch = edge[:, rows[:, None, None], columns[None, :, None], depths[None, None, :]]
    image_patch = image[:, rows[:, None, None], columns[None, :, None], depths[None, None, :]]
    x_pos, y_pos, z_pos = rows, columns, depths

    # patch downsampling
    if is_multiresol:
        edge_patch = F.interpolate(edge_patch.unsqueeze(0), size=(patch_size, patch_size, patch_size), mode='trilinear', align_corners=True).squeeze(0)
        edge_patch = torch.where(edge_patch >= 0.4, torch.tensor(1.0), torch.tensor(0.0))
        image_patch = F.interpolate(image_patch.unsqueeze(0), size=(patch_size, patch_size, patch_size), mode='trilinear', align_corners=True).squeeze(0)

        x_pos = F.interpolate(x_pos.unsqueeze(0).unsqueeze(0).float(), size=patch_size, mode='linear', align_corners=True).squeeze()
        y_pos = F.interpolate(y_pos.unsqueeze(0).unsqueeze(0).float(), size=patch_size, mode='linear', align_corners=True).squeeze()
        z_pos = F.interpolate(z_pos.unsqueeze(0).unsqueeze(0).float(), size=patch_size, mode='linear', align_corners=True).squeeze()

    x_pos = x_pos.view(-1, 1, 1).repeat(1, patch_size, patch_size)
    y_pos = y_pos.view(1, -1, 1).repeat(patch_size, 1, patch_size)
    z_pos = z_pos.view(1, 1, -1).repeat(patch_size, patch_size, 1)

    # Normalization to range [-1, 1]
    x_pos = (x_pos / (h - 1) - 0.5) * 2.
    y_pos = (y_pos / (w - 1) - 0.5) * 2.
    z_pos = (z_pos / (d - 1) - 0.5) * 2.

    data_pos = torch.stack((x_pos, y_pos, z_pos), dim=0)

    return edge_patch, image_patch, data_pos

def pad(image, target_size):
    _, orig_height, orig_width, orig_depth = image.shape

    pad_h = max(target_size[0] - orig_height, 0)
    pad_w = max(target_size[1] - orig_width, 0)
    pad_d = max(target_size[2] - orig_depth, 0)

    pad_h = [pad_h // 2, pad_h - pad_h // 2]
    pad_w = [pad_w // 2, pad_w - pad_w // 2]
    pad_d = [pad_d // 2, pad_d - pad_d // 2]

    padded_image = torch.zeros(
        (image.size(0), target_size[0], target_size[1], target_size[2]),  # (C, H, W, D)
        dtype=image.dtype,
        device=image.device
    )

    padded_image[:, 
        pad_h[0]:pad_h[0] + orig_height,
        pad_w[0]:pad_w[0] + orig_width,
        pad_d[0]:pad_d[0] + orig_depth
    ] = image

    return padded_image

def crop(image, target_size):
    _, orig_height, orig_width, orig_depth = image.shape

    d_start = (orig_depth - target_size[2]) // 2 if orig_depth > target_size[2] else 0
    h_start = (orig_height - target_size[0]) // 2 if orig_height > target_size[0] else 0
    w_start = (orig_width - target_size[1]) // 2 if orig_width > target_size[1] else 0

    cropped_image = image[
        :,  # 모든 채널 유지
        h_start:h_start + target_size[0],
        w_start:w_start + target_size[1],
        d_start:d_start + target_size[2]
    ]

    return cropped_image

def adjust_to_multiple_of_eight(size):
    return (size // 8 + 1) * 8 if size % 8 != 0 else size

def pad_to_multiple_of_eight(image):
    _, orig_height, orig_width, orig_depth = image.shape

    target_height = adjust_to_multiple_of_eight(orig_height)
    target_width = adjust_to_multiple_of_eight(orig_width)
    target_depth = adjust_to_multiple_of_eight(orig_depth)

    if target_height == orig_height and target_width == orig_width and target_depth == orig_depth:
        return image

    padded_image = pad(image, [target_height, target_width, target_depth])
    return padded_image

def augment(edge, image):
    if len(edge.shape) == 3:
        axis = random.choice([1, 2])
    elif len(edge.shape) == 4:
        axis = random.choice([1, 2, 3])
    edge = torch.flip(edge, dims=[axis])
    image = torch.flip(image, dims=[axis])
    return edge, image

class Harmony2DDataset(Dataset):
    def __init__(self, edge_path, image_path, patch_size=None, subject_id=None, is_train=True, is_multiresol=False, transform=None):

        self.edge_path = edge_path
        self.image_path = image_path

        self.is_train = is_train
        self.is_multiresol = is_multiresol

        self.patch_size = patch_size
        self.patchify = multiresol_patchify2D

        with open(self.image_path, 'rb') as file:
            self.image_data = pickle.load(file)

        with open(self.edge_path, 'rb') as file:
            self.edge_data = pickle.load(file)
            self.num_subjects = len(self.edge_data)
            self.num_slices = self.edge_data[0].shape[-1]

        # 특정 서브젝트만 사용할 경우
        self.subject_id = subject_id
        self.total_images = self.num_subjects * self.num_slices
        if subject_id is not None:
            assert subject_id < self.num_subjects, f"Invalid subject_id {subject_id}. Max is {self.num_subjects - 1}."
            self.subject_id = subject_id
            self.total_images = self.num_slices

        self.transform = None
        if transform is None and is_train is True:
            self.transform = augment

    def __len__(self):
        return self.total_images

    def load_single_slice_image(self, subject_idx, slice_idx):
        volume_data = self.image_data[subject_idx]
        slice_data = volume_data[:, :, slice_idx]
        slice_data = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0)
        return slice_data
    
    def load_single_slice_edge(self, subject_idx, slice_idx):
        volume_data = self.edge_data[subject_idx]
        slice_data = volume_data[:, :, slice_idx]
        slice_data = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0)
        return slice_data

    def __getitem__(self, idx):
        if self.subject_id is not None:
            subject_idx = self.subject_id
            slice_idx = idx
        else:
            subject_idx = idx // self.num_slices
            slice_idx = idx % self.num_slices

        edge_slice = self.load_single_slice_edge(subject_idx, slice_idx)
        image_slice = self.load_single_slice_image(subject_idx, slice_idx)

        if self.transform is not None:
            edge_slice, image_slice = self.transform(edge_slice, image_slice)
        
        if self.patch_size is None:
            return edge_slice, image_slice
        
        else:
            if self.is_train:
                edge_slice, image_slice, data_pos = self.patchify(edge_slice, image_slice, self.patch_size, is_multiresol=self.is_multiresol)
            else:
                _, h, w = edge_slice.shape
                x_pos = torch.arange(h).view(-1, 1).repeat(1, w)
                y_pos = torch.arange(w).view(1, -1).repeat(h, 1)
                x_pos = (x_pos / (h - 1) - 0.5) * 2.
                y_pos = (y_pos / (w - 1) - 0.5) * 2.
                data_pos = torch.stack([x_pos, y_pos], dim=0)

            return edge_slice, image_slice, data_pos


class Harmony3DDataset(Dataset):
    def __init__(self, edge_path, image_path, patch_size=None, subject_id=None, is_train=True, is_multiresol=False, transform=None):

        self.edge_path = edge_path
        self.image_path = image_path

        self.is_train = is_train
        self.is_multiresol = is_multiresol
        
        self.patch_size = patch_size
        self.patchify = multiresol_patchify3D

        with open(self.image_path, 'rb') as file:
            self.image_data = pickle.load(file)

        with open(self.edge_path, 'rb') as file:
            self.edge_data = pickle.load(file)
            self.num_subjects = len(self.edge_data)

        # 특정 서브젝트만 사용할 경우
        self.subject_id = subject_id
        if subject_id is not None:
            assert subject_id < self.num_subjects, f"Invalid subject_id {subject_id}. Max is {self.num_subjects - 1}."
            self.subject_id = subject_id
            self.num_subjects = 1
        
        self.transform = None
        if transform is None and is_train is True:
            self.transform = augment

    def __len__(self):
        return self.num_subjects

    def load_full_volume_edge(self, subject_idx):
        volume_data = self.edge_data[subject_idx]
        volume_data = torch.tensor(volume_data, dtype=torch.float32).unsqueeze(0) # (1, H, W, D)
        return volume_data
    
    def load_full_volume_image(self, subject_idx):
        volume_data = self.image_data[subject_idx]
        volume_data = torch.tensor(volume_data, dtype=torch.float32).unsqueeze(0) # (1, H, W, D)
        return volume_data
    
    def __getitem__(self, idx):
        if self.subject_id is not None:
            subject_idx = self.subject_id
        else:
            subject_idx = idx

        edge_volume = self.load_full_volume_edge(subject_idx)
        image_volume = self.load_full_volume_image(subject_idx)

        edge_volume = pad_to_multiple_of_eight(edge_volume)
        image_volume = pad_to_multiple_of_eight(image_volume)

        if self.transform is not None:
            edge_volume, image_volume = self.transform(edge_volume, image_volume)
        
        if self.patch_size is None:
            return edge_volume, image_volume
        
        else:
            if self.is_train:
                edge_volume, image_volume, data_pos = self.patchify(edge_volume, image_volume, self.patch_size, is_multiresol=self.is_multiresol)
            else:
                _, h, w, d = edge_volume.shape
                x_pos = torch.arange(h).view(-1, 1, 1).repeat(1, w, d)
                y_pos = torch.arange(w).view(1, -1, 1).repeat(h, 1, d) 
                z_pos = torch.arange(d).view(1, 1, -1).repeat(h, w, 1)

                x_pos = (x_pos / (h - 1) - 0.5) * 2.
                y_pos = (y_pos / (w - 1) - 0.5) * 2.
                z_pos = (z_pos / (d - 1) - 0.5) * 2.

                data_pos = torch.stack([x_pos, y_pos, z_pos], dim=0)

            return edge_volume, image_volume, data_pos
        
class HarmonyDataset(Dataset):
    def __init__(self, dims, *args, **kwargs):
        if dims == 2:
            self.dataset = Harmony2DDataset(*args, **kwargs)
        elif dims == 3:
            self.dataset = Harmony3DDataset(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported dims: {dims}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]