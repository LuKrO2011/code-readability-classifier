from pathlib import Path

import torch

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.structural_extractor import (
    StructuralExtractor,
)
from readability_classifier.models.extractors.visual_extractor import VisualExtractor
from readability_classifier.utils.config import BaseModelConfig, ViStModelInput


class ViStModelConfig(BaseModelConfig):
    """
    The config for the VisualModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        super().__init__(**kwargs)
        self.input_length = kwargs.get("input_length", 15616)
        self.output_length = kwargs.get("output_length", 1)
        self.dropout = kwargs.get("dropout", 0.5)


class ViStModel(BaseModel):
    """
    A code readability model based on the visual and structural features of a code
    snippet.
    The model consists of a visual feature extractor, a structural feature extractor and
    multiple dense layers.
    The input consists of an image and a character matrix. The output is a vector of
    size 1.
    """

    def __init__(self, config: ViStModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()

        # Feature extractors
        self.visual_extractor = VisualExtractor.build_from_config()
        self.structural_extractor = StructuralExtractor.build_from_config()

        # Define own layers
        self._build_classification_layers(config)

    def forward(self, x: ViStModelInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model containing the image.
        :return: The output of the model.
        """
        # Feature extractors
        visual_features = self.visual_extractor(x.image)
        structural_features = self.structural_extractor(x.matrix)

        # Concatenate features
        features = torch.cat((visual_features, structural_features), dim=1)

        # Pass through dense layers
        return self._forward_classification_layers(features)

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(ViStModelConfig(**params))
