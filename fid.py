import glob

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchmetrics.image.fid import FrechetInceptionDistance

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- config ---
fake_dir = "results"      # folder containing your generated images
batch_size = 128

# normalize=True -> metric expects float images in [0, 1]
fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

# --- 1. fake images: load from your output folder ---
paths = sorted(glob.glob(f"{fake_dir}/*.png"))
if not paths:
    raise FileNotFoundError(f"no .png files found in ./{fake_dir}/")

for i in range(0, len(paths), batch_size):
    batch = paths[i:i + batch_size]
    # read_image returns uint8 [C,H,W] in [0,255]; stack and scale to [0,1]
    imgs = torch.stack([read_image(p)[:3] for p in batch]).float() / 255.0
    fid.update(imgs.to(device), real=False)
print(f"fed {len(paths)} fake images")

# --- 2. real images: same COUNT, plain [0,1] (NOT the [-1,1] training transform) ---
real_ds = datasets.CIFAR10(
    "./data", train=True, download=True, transform=transforms.ToTensor()
)
real_loader = DataLoader(real_ds, batch_size=batch_size)

seen = 0
for imgs, _ in real_loader:
    if seen >= len(paths):
        break
    fid.update(imgs.to(device), real=True)
    seen += imgs.size(0)
print(f"fed {seen} real images")

# --- 3. score ---
print("FID:", fid.compute().item())
