import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from unet import UNet
from diffusion import DDPM

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channel=3, t_dim=128).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

ddpm = DDPM(model).to(device)

imgs = ddpm.sample(num_images=225, device=device)
imgs = (imgs.clamp(-1,1)+1)/2  # [0,1] for saving
save_image(imgs, "samples_225.png", nrow=15)
print("saved samples_225.png")
