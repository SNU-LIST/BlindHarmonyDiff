import torch
import os
from rf_img2img import RectifiedFlow, Unet, Trainer
from dataset.harmony_dataset import HarmonyDataset
import random
import numpy as np

# Set GPU devices (all GPUs will be available without manual memory limitation)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPUs 0, 1 are available

results_folder = './experiments/3d_n_patch_02'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(0)

# Define datasets and model
dims = 3
in_channels = 1
out_channels = 1
patch_size = 64
batch_size = 8
image_size = (144, 208, 92)

train_dataset = HarmonyDataset(
    dims=dims,
    edge_path='/fast_storage/hayeon/pckl_blindharmony_new/pkl_edge_35177_train.pklv4',
    image_path='/fast_storage/hayeon/pckl_blindharmony_new/pkl_image_35177_train.pklv4',
    # image_size=image_size[:dims],
    patch_size=patch_size,
    is_train=True,
    is_multiresol=True
)

val_dataset = HarmonyDataset(
    dims=dims,
    edge_path='/fast_storage/hayeon/pckl_blindharmony_new/pkl_edge_35177_val.pklv4',
    image_path='/fast_storage/hayeon/pckl_blindharmony_new/pkl_image_35177_val.pklv4',
    # image_size=image_size[:dims],
    patch_size=patch_size,
    subject_id=0,
    is_train=False # memory issue
)

# Adjust input channels for patch coordinate info
in_channels = in_channels * 2
if patch_size is not None:
    in_channels += dims

# Initialize the model
model = Unet(dims=dims, channels=in_channels, out_dim=out_channels, dim=64)

# Use DataParallel to distribute the model across all available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)  # No manual memory limit, automatic GPU usage
model = model.to(device)

# Rectified flow initialization
rectified_flow = RectifiedFlow(model)

# Initialize trainer
trainer = Trainer(
    rectified_flow,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=5e-5,  # 2e-4 was the original
    batch_size=batch_size,
    num_train_steps=300_001,
    validation_every=10000,
    checkpoint_every=10000,
    results_folder=results_folder,
    log_loss_every=100,
    resume_checkpoint='/fast_storage/hayeon/RF/experiments/3d_n_patch_02/checkpoints/checkpoint_100000.pt',
    start_step=100001
)

# Run the trainer
trainer()
