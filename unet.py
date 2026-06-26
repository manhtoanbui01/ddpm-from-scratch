import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
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
        x = self.conv1(F.silu(self.gr_norm1(x)))
        x = x + self.t_proj(F.silu(t_embed))[:, :, None, None]
        x = self.conv2(self.dropout(F.silu(self.gr_norm2(x))))
        return self.skip(identity) + x
        
class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 3, 2, 1)
    
    def forward(self, x):
        return self.down(x)
        

class Encoder(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channel, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(128, 128, t_dim=512)
        self.down1 = DownSample(128)
        self.res2 = ResidualBlock(128, 256, t_dim=512)
        self.down2 = DownSample(256)
        self.res3 = ResidualBlock(256, 256, t_dim=512)
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
    def __init__(self, in_channel: int):
        super().__init__()
        self.up3 = UpSample(in_channel)
        self.res3 = ResidualBlock(in_channel*2, 256, t_dim=512)
        
        self.up2 = UpSample(256)
        self.res2 = ResidualBlock(256*2, 256, t_dim=512)
        
        self.up1 = UpSample(256)
        self.res1 = ResidualBlock(256+128, 128, t_dim=512)
        
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
    def __init__(self, in_channel: int):
        super().__init__()
        
        
class BottleNeck(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.res_in = ResidualBlock(in_channel, in_channel, t_dim=512)
        self.res_out = ResidualBlock(in_channel, in_channel, t_dim=512)
             


class UNet(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder
    