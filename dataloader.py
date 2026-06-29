import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_dataloaders(
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        image_size=32,
):
    # Scale pixel from [0,1] to [-1,1] - match the diffusion prior N(0,I)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # No flip augmentation for the test/eval set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)        # [128, 3, 32, 32]
    print("Value range:", images.min().item(), "to", images.max().item())  # ~ -1 to 1
    print("Num train batches:", len(train_loader))
    print("Num test batches:", len(test_loader))
