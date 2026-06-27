import torch.nn as nn

linear = nn.Linear(64, 128)

print(linear.weight.shape)   # torch.Size([128, 64])
print(linear.bias.shape)     # torch.Size([128])

print(sum(p.numel() for p in linear.parameters()))
# 8320