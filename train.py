import torch

from unet import UNet
from dataloader import get_cifar10_dataloaders

train_loader, test_loader = get_cifar10_dataloaders()

