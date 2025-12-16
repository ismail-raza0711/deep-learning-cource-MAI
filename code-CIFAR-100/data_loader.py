"""
Data loading and preprocessing for CIFAR-100.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import Config


class ApplyTransform(torch.utils.data.Dataset):
    """
    A simple wrapper that applies a transform to a dataset subset
    without affecting the parent dataset.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_data_loaders(
    data_dir="./data", batch_size=128, val_split=0.1, num_workers=2, model_type="cnn"
):
    # CIFAR-100 normalization constants (mean and std per channel)
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    # Data transforms
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    # Data augmentation for training set
    if model_type.lower() == "cnn":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.RandomRotation(15),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
                ),
            ]
        )
    else:
        # Simplified augmentation for MLP
        train_transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    # Load full training dataset
    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True)

    # Split into train and validation
    val_size = int(len(full_train_dataset) * val_split)
    # train_size = len(full_train_dataset) - val_size
    train_indices, val_indices = random_split(
        full_train_dataset, [len(full_train_dataset) - val_size, val_size]
    )

    """train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2025),  # For reproducibility
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = data_transform"""

    # Load test dataset
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=data_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        ApplyTransform(train_indices, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # !!! Changed to False to avoid potential issues on some systems
    )

    val_loader = DataLoader(
        ApplyTransform(val_indices, transform=data_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # !!! Changed to False to avoid potential issues on some systems
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # !!! Changed to False to avoid potential issues on some systems
    )

    return train_loader, val_loader, test_loader


def get_cifar100_classes():
    return datasets.CIFAR100(root=Config.DATA_DIR, download=False).classes


if __name__ == "__main__":
    # Test the data loader
    print("Loading CIFAR-100 data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=64, model_type=Config.MODEL_TYPE
    )

    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print(f"\nNumber of batches:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")

    print(f"\nClasses: {get_cifar100_classes()}")
