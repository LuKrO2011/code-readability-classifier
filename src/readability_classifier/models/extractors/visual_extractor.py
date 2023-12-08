from pathlib import Path

import torch
from torch import nn as nn

from readability_classifier.utils.config import VisualInput
from src.readability_classifier.models.base_model import BaseModel


class VisualExtractorConfig:
    """
    The config for the VisualExtractor.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        pass


# TODO: What if without padding or stride specified?
class VisualExtractor(BaseModel):
    """
    A visual extractor model. Also known as ImageExtractor.
    The model consists of multiple alternating 2D convolution and max-pooling layers
    plus a flatten layer.
    The input is an image of size (3, 128, 128) and the output is a vector of size 6400.
    """

    def __init__(self, config: VisualExtractorConfig) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        # Alternating 2D convolution and max-pooling layers

        # In code: kernel_size=3, in code: padding = 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2)
        self.relu = nn.ReLU()

        # In paper: stride not specified
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # In code: kernel_size=3, in code: padding = 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)

        # In paper: stride not specified
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # In paper: padding not specified
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # In paper: stride not specified
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Same as in paper
        self.flatten = nn.Flatten()

    def forward(self, x: VisualInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input tensor, which is an image.
        :return: The output tensor.
        """
        x = x.image

        # Apply convolutional and pooling layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output of the conv layers
        x = self.flatten(x)

        return x

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(VisualExtractorConfig(**params))
