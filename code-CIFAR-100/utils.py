"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import math


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        average_loss, accuracy
    """

    """ 
    !!!!!!!!!!!!!
    Set model to training mode
    !!!!!!!!!!!!!!
    """
    model.train()  # this line was missing

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        """ 
        !!!!!!!!!!!!!
        Gradients accumulate across batches.
        Clear gradients before backward pass
        !!!!!!!!!!!!!!
        """
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        average_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def plot_training_history(
    train_losses, val_losses, train_accs, val_accs, save_path=None
):
    """
    Plot training and validation losses and accuracies.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()


def save_checkpoint(
    model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, filepath
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        train_acc: Training accuracy
        val_acc: Validation accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load to

    Returns:
        epoch, train_loss, val_loss, train_acc, val_acc
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {checkpoint['epoch']}")

    return (
        checkpoint["epoch"],
        checkpoint["train_loss"],
        checkpoint["val_loss"],
        checkpoint["train_acc"],
        checkpoint["val_acc"],
    )


# =========================================================================
mean = np.array([0.5071, 0.4865, 0.4409])
std = np.array([0.2673, 0.2564, 0.2762])


def get_dataset():
    # CIFAR-100 normalization
    mean = np.array([0.5071, 0.4865, 0.4409])
    std = np.array([0.2673, 0.2564, 0.2762])

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    return dataset


def plot_dataset_samples(
    num_samples=30, images_per_row=5, random_samples=True, train=True
):
    dataset = get_dataset()

    class_names = dataset.classes

    if random_samples:
        indices = torch.randperm(len(dataset))[:num_samples]
    else:
        indices = range(num_samples)

    # ---------- GRID SHAPE ----------
    rows = math.ceil(num_samples / images_per_row)
    cols = images_per_row

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))

    axes = axes.flatten()  # makes indexing easy

    for ax, idx in zip(axes, indices):
        image, label = dataset[idx]

        # Tensor → numpy
        image = image.permute(1, 2, 0).numpy()

        # Unnormalize
        image = std * image + mean
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")

    # Turn off unused subplots
    for ax in axes[len(indices) :]:
        ax.axis("off")

    title = "CIFAR-100 Training Samples" if train else "CIFAR-100 Test Samples"
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_cifar100_class_distribution(save_path=None):
    dataset = get_dataset()
    class_names, class_counts = get_cifar100_class_distribution(dataset)

    plt.figure(figsize=(20, 6))
    plt.bar(range(len(class_counts)), class_counts.numpy())
    plt.xlabel("Class index")
    plt.ylabel("Number of samples")
    plt.title("CIFAR-100 Class Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def extract_labels_from_dataset(dataset):
    return torch.tensor(dataset.targets)


def get_cifar100_class_distribution(dataset):
    labels = torch.tensor(dataset.targets)
    num_classes = 100
    class_names = dataset.classes
    class_counts = torch.bincount(labels, minlength=num_classes)
    return class_names, class_counts


if __name__ == "__main__":
    print("Utility functions loaded successfully.")
    print("Available functions:")
    print("  - train_one_epoch()")
    print("  - evaluate()")
    print("  - plot_training_history()")
    print("  - save_checkpoint()")
    print("  - load_checkpoint()")
