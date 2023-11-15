from pathlib import Path

import torch
from torch import nn as nn

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.structural_extractor import (
    StructuralExtractor,
)


class StructuralModelConfig:
    """
    The config for the StructuralModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        self.input_length = kwargs.get("input_length", 41472)
        self.output_length = kwargs.get("output_length", 1)
        self.dropout = kwargs.get("dropout", 0.5)


class StructuralModel(BaseModel):
    """
    A code readability model based on the structural features of the code.
    The model consists of a structural feature extractor plus own layers.
    The input consists of a character matrix. The character matrix is of size
    (305, 50). The output is a single value representing the readability of the code
    snippet.
    The own layers consist of:
    1. Dense layer
    2. ReLU layer
    3. Dropout layer
    4. Dense layer
    5. ReLU layer
    6. Dense layer
    7. Sigmoid layer
    """

    def __init__(self, config: StructuralModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()

        # Feature extractors
        self.structural_extractor = StructuralExtractor.build_from_config()

        # Define own layers
        self.dense1 = nn.Linear(config.input_length, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.dense2 = nn.Linear(64, 16)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        character_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param character_matrix: The character matrix tensor.
        :return: The output of the model.
        """
        # Feature extractors
        structural_features = self.structural_extractor(character_matrix)

        # Pass through dense layers
        x = self.dense1(structural_features)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        return x

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(StructuralModelConfig(**params))
