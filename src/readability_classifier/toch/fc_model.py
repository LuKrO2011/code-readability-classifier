import torch
from torch import nn

from src.readability_classifier.utils.config import BaseModelConfig, ViStModelInput


class FullyConnectedModel(nn.Module):
    """
    A fully connected model for code readability classification. The output is a single
    value representing the readability of the code snippet.
    """

    def __init__(self, config: BaseModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()
        self.dense1 = nn.Linear(config.input_length, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.dense2 = nn.Linear(64, 16)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(16, config.output_length)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: ViStModelInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input.
        :return: The output.
        """
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        return self.sigmoid(x)

    # TODO: .to(device) makes tests fail but is necessary to enable input size change
    def update_input_length(self, input_length: int, device: torch.device) -> None:
        """
        Update the input length of the forward classification layers, if needed.
        :param input_length: The new input length.
        :param device: The device to put the new layer on.
        :return: The output.
        """
        if input_length != self.dense1.in_features:
            self.dense1 = nn.Linear(input_length, 64)
            self.dense1 = self.dense1.to(device)
