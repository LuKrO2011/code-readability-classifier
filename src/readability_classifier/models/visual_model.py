from pathlib import Path

import torch

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.visual_extractor import VisualExtractor
from readability_classifier.utils.config import BaseModelConfig, VisualInput


class VisualModelConfig(BaseModelConfig):
    """
    The config for the VisualModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        super().__init__(**kwargs)
        self.input_length = kwargs.get("input_length", 12544)
        self.output_length = kwargs.get("output_length", 1)
        self.dropout = kwargs.get("dropout", 0.5)


class VisualModel(BaseModel):
    """
    A code readability model based on the visual features of the code.
    The model consists of a visual feature extractor plus own layers.
    The input consists of an RGB image. The image is a tensor of size (3, 128, 128).
    The output is a single value representing the readability of the code snippet.
    """

    def __init__(self, config: VisualModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()

        # Feature extractors
        self.visual_extractor = VisualExtractor.build_from_config()

        # Define own layers
        self._build_classification_layers(config)

    def forward(self, x: VisualInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model containing the image.
        :return: The output of the model.
        """
        # Feature extractors
        visual_features = self.visual_extractor(x.image)

        # Pass through dense layers
        return self._forward_classification_layers(visual_features)

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(VisualModelConfig(**params))
