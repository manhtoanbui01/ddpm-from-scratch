import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
    
    def _extract(self, buf, t):  # gather and reshape to [B, 1, 1, 1]
        B = t.shape[0]
        out = buf.gather(dim=0, index=t)  # [B]
        return out.reshape(B, 1, 1, 1)

        
    def q_sample(self, x0, t, noise):
        a = self._extract(self.sqrt_alphas_cumprod, t)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t)
        return a * x0 + b * noise
    
    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        loss = F.mse_loss(pred, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self._extract(self.betas, t)
        alphas_t = self._extract(self.alphas, t)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t)
        
        noise = torch.randn_like(x)
        eps = self.model(x, t)
        
        mean = 1.0 / (torch.sqrt(alphas_t)) * (x - (betas_t/sqrt_one_minus_alphas_cumprod_t) * eps)
        
        if t[0] == 0:  # last step, no noise inject
            return mean
        else: 
            return mean + betas_t.sqrt() * noise  # sampling step, \sigma_t = sqrt(beta_t)
    
    @torch.no_grad()
    def sample(self, num_images: int, img_size=32, channels=3, device="cuda"):
        x = torch.randn(num_images, channels, img_size, img_size, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_images,), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x
    
    

    