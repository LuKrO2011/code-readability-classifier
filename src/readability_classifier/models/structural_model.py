from pathlib import Path

import torch
from torch import nn as nn

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.structural_extractor import (
    StructuralExtractor,
)
from readability_classifier.utils.config import BaseModelConfig, StructuralInput


class StructuralModelConfig(BaseModelConfig):
    """
    The config for the StructuralModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        super().__init__(**kwargs)
        self.input_length = kwargs.get("input_length", 9216)
        self.output_length = kwargs.get("output_length", 1)
        self.dropout = kwargs.get("dropout", 0.5)


class StructuralModel(BaseModel):
    """
    A code readability model based on the structural features of the code.
    The model consists of a structural feature extractor plus own layers.
    The input consists of a character matrix. The character matrix is of size
    (305, 50). The output is a single value representing the readability of the code
    snippet.
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
        self._build_classification_layers(config)

    def forward(self, x: StructuralInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model containing the character matrix.
        :return: The output of the model.
        """
        # Feature extractors
        structural_features = self.structural_extractor(x.character_matrix)

        # Pass through dense layers
        return self._forward_classification_layers(structural_features)

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(StructuralModelConfig(**params))
