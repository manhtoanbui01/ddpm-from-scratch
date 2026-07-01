import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from unet import UNet
from diffusion import DDPM
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channel=3, t_dim=128).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

ddpm = DDPM(model).to(device)

# save image for visualize
imgs = ddpm.sample(num_images=225, device=device)
imgs = (imgs.clamp(-1,1)+1)/2  # [0,1] for saving
save_image(imgs, "samples.png", nrow=15)
print("saved samples_225.png")


# save inference images
# num_images = 50000
# batch_size = 500
# saved = 0
# output_path = "results"
# os.makedirs(output_path, exist_ok=True)
# while saved < num_images:
#     cur = min(batch_size, num_images-saved)
#     imgs = ddpm.sample(num_images=cur, device=device)
#     imgs = (imgs.clamp(-1,1) + 1)/2
#     for img in imgs:
#         save_image(img, f"results/{saved:05d}.png")
#         saved+=1
