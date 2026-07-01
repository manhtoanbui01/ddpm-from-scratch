import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# load CIFAR-10 in plain [0,1]
ds = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=512, num_workers=4)

# running sum of all images -> mean
total = torch.zeros(3, 32, 32)
n = 0
for imgs, _ in loader:
    total += imgs.sum(dim=0)   # sum over the batch
    n += imgs.size(0)

mean_img = total / n           # [3,32,32], the average of all training images
print(f"averaged {n} images | pixel range {mean_img.min():.3f}..{mean_img.max():.3f}")

save_image(mean_img, "average_training_image.png")
print("saved average_training_image.png")
