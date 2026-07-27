import torch
import os
from torchvision.utils import save_image

from unet import UNet
from diffusion import DDPM

class DDIM(DDPM):
    @torch.no_grad()
    def p_sample(self, x, t, t_prev):
        if t_prev[0] == 0:  # the last denoising step, not have t_prev
            somab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t)  # somab: sqrt_one_minus_alpha_bar
            sab_t = self._extract(self.sqrt_alphas_cumprod, t) 
            
            eps = self.model(x, t)
            
            x0 = (x - somab_t * eps) / sab_t
            return x0
        else:
            somab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t)  # somab: sqrt_one_minus_alpha_bar
            somab_prev = self._extract(self.sqrt_one_minus_alphas_cumprod, t_prev)
            sab_t = self._extract(self.sqrt_alphas_cumprod, t) 
            sab_prev = self._extract(self.sqrt_alphas_cumprod, t_prev)
            
            eps = self.model(x, t)
            
            x_prev = sab_prev / sab_t * (x - somab_t * eps) + somab_prev * eps
            return x_prev
            
    @torch.no_grad()
    def sample(self, num_images: int, num_inference_steps=50, img_size=32, channels=3, device="cuda"):
        x = torch.randn(num_images, channels, img_size, img_size, device=device)
        ts = torch.linspace(0, self.timesteps - 1, num_inference_steps).round().long()  # [0..999]
        inference_steps = ts.flip(0)   # descending: 999, ..., 0
        
        for i in range(num_inference_steps - 1):
            t = torch.full((num_images,), inference_steps[i], dtype=torch.long, device=device)
            t_prev = torch.full((num_images,), inference_steps[i+1], dtype=torch.long, device=device)
            x = self.p_sample(x, t, t_prev)
        return x
            
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channel=3, t_dim=128).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

ddim = DDIM(model).to(device)

imgs = ddim.sample(num_images=225, num_inference_steps=50, device=device)
imgs = (imgs.clamp(-1,1)+1)/2  # [0,1] for saving
save_image(imgs, "samples_ddim.png", nrow=15)
print("save samples_ddim.png")


