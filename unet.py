import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    The core building block of the U-Net. It does two things:
      1. Transforms features with two conv layers (GroupNorm -> SiLU -> Conv), 
        The ResBlock uses pre-activation (GroupNorm → SiLU → Conv), the main benefit is that it keeps the residual skip path a clean identity: the block computes output = x + conv(SiLU(norm(x))), with nothing applied after the addition. This gives an unobstructed gradient highway through the skip connection (better gradient flow in deep nets) 
        Pre-activation also keeps the Conv layer's input stable, since it is normalized (and activated) before the Conv.
        This advantage exists because of the residual connection—without a skip path, the ordering matters far less.
      2. Injects the timestep information so the block knows the noise level.
    A residual (skip) connection adds the input back to the output, which helps
    gradients flow and lets the block learn only the *change* it needs to make.
    """
    def __init__(self, in_channel, out_channel, t_dim, num_groups=32, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.gr_norm1 = nn.GroupNorm(num_groups, num_channels=in_channel)
        self.gr_norm2 = nn.GroupNorm(num_groups, num_channels=out_channel)
        self.dropout = nn.Dropout(dropout)
        self.t_proj = nn.Linear(t_dim, out_channel)
        
        if in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_embed):
        identity = x
        # First sub-block: normalize -> activate -> convolve.  [B, out_channel, H, W]
        x = self.conv1(F.silu(self.gr_norm1(x)))

        # Inject time: project t_embed to [B, out_channel], then reshape to [B, out_channel, 1, 1] so it broadcasts and adds the same bias to every pixel (The noise level is also the same everywhere in the image, but different bias per channel)
        x = x + self.t_proj(F.silu(t_embed))[:, :, None, None]

        # Second sub-block: normalize -> activate -> dropout -> convolve.
        x = self.conv2(self.dropout(F.silu(self.gr_norm2(x))))
        return self.skip(identity) + x
        
class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 3, 2, 1)
    
    def forward(self, x):
        return self.down(x)
        

class Encoder(nn.Module):
    def __init__(self, in_channel: int, t_dim: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channel, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(128, 128, t_dim=t_dim)
        self.down1 = DownSample(128)
        self.res2 = ResidualBlock(128, 256, t_dim=t_dim)
        self.down2 = DownSample(256)
        self.res3 = ResidualBlock(256, 256, t_dim=t_dim)
        self.down3 = DownSample(256)
        
    def forward(self, x, t_embed):  # x: [B, 3, 32, 32]
        x = self.conv_in(x)  # [B, 128, 32, 32]
        
        x = self.res1(x, t_embed)  # [B, 128, 32, 32]
        res1 = x
        x = self.down1(x)  # [B, 128, 16, 16]
        
        x = self.res2(x, t_embed)  # [B, 256, 16, 16]
        res2 = x
        x = self.down2(x)  # [B, 256, 8, 8]
        
        x = self.res3(x, t_embed)  # [B, 256, 8, 8]
        res3 = x
        x = self.down3(x)  # [B, 256, 4, 4]
        
        return x, res1, res2, res3
        
        
class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)  # using this instead of ConvTransposed2d to avoid checkerboard artifacts
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)              


class Decoder(nn.Module):
    def __init__(self, in_channel: int, t_dim: int):
        super().__init__()
        self.up3 = UpSample(in_channel)
        self.res3 = ResidualBlock(in_channel*2, 256, t_dim=t_dim)
        
        self.up2 = UpSample(256)
        self.res2 = ResidualBlock(256*2, 256, t_dim=t_dim)
        
        self.up1 = UpSample(256)
        self.res1 = ResidualBlock(256+128, 128, t_dim=t_dim)
        
        self.gr_norm = nn.GroupNorm(32, 128)
        
        self.conv_out = nn.Conv2d(128, 3, 3, 1, 1)
        
    def forward(self, x, t_embed, res1, res2, res3):
        x = self.up3(x)
        x = torch.concatenate([x, res3], dim=1)
        x = self.res3(x, t_embed)
        
        x = self.up2(x)
        x = torch.concatenate([x, res2], dim=1)
        x = self.res2(x, t_embed)

        x = self.up1(x)
        x = torch.concatenate([x, res1], dim=1)
        x = self.res1(x, t_embed)

        x = self.gr_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x
        
class SelfAttn(nn.Module):
    def __init__(self, in_channel: int, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, 3*in_channel, kernel_size=1)
        self.out = nn.Conv2d(in_channel, in_channel, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        x = self.norm(x)
        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=1)
        q = q.reshape(B, C, H*W)
        k = k.reshape(B, C, H*W)
        v = v.reshape(B, C, H*W)
        attn_score = torch.softmax(q.transpose(1,2) @ k/(C**0.5), dim=-1)  # [B, H*W, H*W], The sum Σ_c is over channels - the channel dimension is contracted (summed away), acting as the feature/embedding dimension along which two pixels' vectors are compared.
        x = (attn_score @ v.transpose(1,2)).transpose(1,2)  # [B, C, H*W]
        x = self.out(x.reshape(B, C, H, W))
        return identity + x

        
        
class BottleNeck(nn.Module):
    def __init__(self, in_channel: int, t_dim: int):
        super().__init__()
        self.res_in = ResidualBlock(in_channel, in_channel, t_dim=t_dim)
        self.self_attn = SelfAttn(in_channel)
        self.res_out = ResidualBlock(in_channel, in_channel, t_dim=t_dim)
    
    def forward(self, x, t_embed):
        x = self.res_in(x, t_embed)
        x = self.self_attn(x)
        x = self.res_out(x, t_embed)
        return x

import math           
def sinusoidal_embedding(t, dim):
    """
    t:   [B]  integer timesteps
    out: [B, dim]
    """
    half = dim // 2
    # frequencies: 1 / 10000^(i/half), geometrically spaced
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )                                          # [half]
    args = t[:, None].float() * freqs[None, :] # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
    return emb


class UNet(nn.Module):
    def __init__(self, in_channel: int, t_dim: int):
        super().__init__()
        self.encoder = Encoder(in_channel=in_channel, t_dim=t_dim)
        self.bottle_neck = BottleNeck(in_channel=256, t_dim=t_dim)
        self.decoder = Decoder(in_channel=256, t_dim=t_dim)
        self.t_dim = t_dim
    
    def forward(self, x, t):
        t_embed = sinusoidal_embedding(t, dim=self.t_dim)
        x, res1, res2, res3 = self.encoder(x, t_embed)
        x = self.bottle_neck(x, t_embed)
        x = self.decoder(x, t_embed, res1, res2, res3)
        return x
