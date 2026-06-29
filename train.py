import torch

from unet import UNet
from diffusion import DDPM
from dataloader import get_cifar10_dataloaders

train_loader, test_loader = get_cifar10_dataloaders()
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 100

model = UNet(in_channel=3, t_dim=128).to(device)
ddpm = DDPM(model).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

step = 0
for epoch in range(epochs):
    for x0, _ in train_loader:   # ignore the CIFAR labels
        x0 = x0.to(device)
        t = torch.randint(0, 1000, (x0.size(0),), device=device)
        loss = ddpm.p_losses(x0, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        step += 1
 
torch.save(model.state_dict(), "model_weighs.pth")
