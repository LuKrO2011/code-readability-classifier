import torch
from torch import nn as nn


# TODO: What if without padding or stride specified?
class VisualExtractor(nn.Module):
    """
    Also known as ImageExtractor. A visual feature extractor for code readability
    classification. The model consists of multiple alternating 2D convolution and
    max-pooling layers plus a flatten layer.
    The input is an image of size (3, 128, 128) and the output is a vector of size 64.
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        # Alternating 2D convolution and max-pooling layers

        # In paper: kernel_size=2, padding not specified
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # In paper: stride not specified
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # In paper: kernel_size=2, padding not specified
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )

        # In paper: stride not specified
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # In paper: padding not specified
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # In paper: stride not specified
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Same as in paper
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input tensor, which is an image.
        :return: The output tensor.
        """
        # Apply convolutional and pooling layers
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool3(x)

        # Flatten the output of the conv layers
        x = self.flatten(x)

        return x
