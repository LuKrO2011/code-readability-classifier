from pathlib import Path

import torch
from torch import nn as nn

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.semantic_extractor import (
    SemanticExtractor,
)
from readability_classifier.models.extractors.structural_extractor import (
    StructuralExtractor,
)
from readability_classifier.models.extractors.visual_extractor import VisualExtractor


class TowardsModelConfig:
    """
    The config for the TowardsModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        self.input_length = kwargs.get("input_length", 58432)
        self.output_length = kwargs.get("output_length", 1)
        self.dropout = kwargs.get("dropout", 0.5)


class TowardsModel(BaseModel):
    """
    A code readability model.
    The model consists of a visual, a semantic and a structural feature extractor plus
    own layers.
    The input consists of a character matrix, a bert encoding and an image of the code.
    In the image, words are replaced by color bars depending on their token type.
    The character matrix is of size (305, 50), the bert encoded code snippet is of size
    512 and the image is of size (3, 128, 128). The output is a single value
    representing the readability of the code snippet.
    The own layers consist of:
    1. Fully connected layer
    2. Dropout layer
    3. Fully connected layer
    4. Fully connected layer
    """

    def __init__(self, config: TowardsModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()

        # Feature extractors
        self.structural_extractor = StructuralExtractor.build_from_config()
        self.semantic_extractor = SemanticExtractor.build_from_config()
        self.visual_extractor = VisualExtractor.build_from_config()

        # Define own layers
        self.dense1 = nn.Linear(config.input_length, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.dense2 = nn.Linear(64, 16)
        self.random_detail = nn.Linear(16, config.output_length)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        character_matrix: torch.Tensor,
        token_input: torch.Tensor,  # Same as input_ids
        segment_input: torch.Tensor,  # Same as token_type_ids
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param image: The image tensor.
        :param token_input: The token input tensor.
        :param segment_input: The segment input tensor.
        :param character_matrix: The character matrix tensor.
        :return: The output of the model.
        """
        # Feature extractors
        structural_features = self.structural_extractor(character_matrix)
        semantic_features = self.semantic_extractor(token_input, segment_input)
        visual_features = self.visual_extractor(image)

        # Concatenate the inputs
        concatenated = torch.cat(
            (structural_features, semantic_features, visual_features), dim=-1
        )

        # Pass through dense layers
        x = self.dense1(concatenated)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.random_detail(x)
        return self.sigmoid(x)

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(TowardsModelConfig(**params))
