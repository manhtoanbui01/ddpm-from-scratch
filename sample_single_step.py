import torch

from torchvision.utils import save_image
from unet import UNet
from diffusion import DDPM

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channel=3, t_dim=128).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

ddpm = DDPM(model).to(device)


@torch.no_grad()
def single_step_sample(ddpm, num_images, img_size=32, channels=3, device="cuda"):
    """Generate images in ONE step: predict noise from pure noise, then jump
    straight to the x0 estimate via the closed form. Produces blurry images
    (the average E[x0|x_T]) -- this is why real DDPM needs many steps."""
    T = ddpm.timesteps
    x_T = torch.randn(num_images, channels, img_size, img_size, device=device)
    t = torch.full((num_images,), T - 1, device=device, dtype=torch.long)  # start at pure noise

    eps = ddpm.model(x_T, t)                                      # single forward pass

    sqrt_ab = ddpm._extract(ddpm.sqrt_alphas_cumprod, t)          # sqrt alpha_bar: [B,1,1,1]
    sqrt_omab = ddpm._extract(ddpm.sqrt_one_minus_alphas_cumprod, t)
    x0 = (x_T - sqrt_omab * eps) / sqrt_ab                        # closed-form jump to x0
    return x0


# save image for visualize
imgs = single_step_sample(ddpm, num_images=225, device=device)
imgs = (imgs.clamp(-1, 1) + 1) / 2  # [0,1] for saving
save_image(imgs, "samples_225_single_step.png", nrow=15)
print("saved samples_225_single_step.png")
