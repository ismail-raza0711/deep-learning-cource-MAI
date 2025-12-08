"""
Network architectures for CIFAR-10 classification.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for CIFAR-10 classification.
    Input: 3x32x32 image tensors flattened to vectors
    Output: 10 class logits
    """

    """
    !!!!!!!!!!!!!
    Wroing input size!
    CIFAR-10 images are of size 3x32x32 = 3072 so input size
    can not be 1024
    !!!!!!!!!!!!!!
    """

    def __init__(
        self, input_size=3072, hidden_sizes=[512, 256], num_classes=10, dropout=0.3
    ):
        super(MLP, self).__init__()

        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification.
    Architecture: Two Conv-BN-ReLU-Conv-BN-ReLU-MaxPool blocks followed by FC layers
    """

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1:
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2:
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        """
        !!!!!!!!!!!!!
        Input: 3 * 32 * 32
        Block 1 MaxPool → 32 channels= 16*16
        Block 2 MaxPool → 64 channels= 8*8
        So flattened size = 64 * 8 * 8 = 4096
        """
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 512),  # fixed input size
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def get_model(model_type="cnn", **kwargs):
    """
    Factory function to create models.

    Args:
        model_type: 'mlp' or 'cnn'
        **kwargs: Additional arguments for model constructor

    Returns:
        PyTorch model
    """
    if model_type.lower() == "mlp":
        return MLP(**kwargs)
    elif model_type.lower() == "cnn":
        return CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'mlp' or 'cnn'.")


if __name__ == "__main__":
    # Test the models
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)

    print("Testing MLP:")
    mlp = MLP()
    output = mlp(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mlp.parameters()):,}")

    print("\nTesting CNN:")
    cnn = CNN()
    output = cnn(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters()):,}")
