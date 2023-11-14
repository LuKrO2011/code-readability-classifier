import torch
from torch import nn as nn


class StructuralExtractor(nn.Module):
    """
    A structural feature extractor for code readability classification. The model
    consists of alternating 2D convolution and max-pooling layers plus a flatten layer.
    The input is a tensor of size (1, 350, 50) and the output is a vector of size 128.
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        # Alternating 2D convolution and max-pooling layers

        # In paper: kernel_size=2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        # Same as in paper
        self.relu = nn.ReLU()

        # In paper: stride not specified
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # In paper: kernel_size=2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

        # In paper: stride not specified
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Same as in paper
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # In paper: stride not specified
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Same as in paper
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input tensor.
        :return: The output tensor.
        """
        # Apply convolutional and pooling layers
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)

        # Flatten the output of the conv layers
        x = self.flatten(x)

        return x
