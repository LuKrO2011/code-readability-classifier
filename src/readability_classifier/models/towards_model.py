from pathlib import Path

import torch

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.structural_extractor import (
    StructuralExtractor,
)
from readability_classifier.models.extractors.visual_extractor import VisualExtractor
from readability_classifier.models.semantic_model import SemanticExtractorEnum
from readability_classifier.utils.config import BaseModelConfig, TowardsInput

EXTRACTOR = SemanticExtractorEnum.OWN_SEGMENT_IDS


class TowardsModelConfig(BaseModelConfig):
    """
    The config for the TowardsModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        super().__init__(**kwargs)
        self.input_length = kwargs.get("input_length", 1)  # Overwritten as needed
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
    """

    def __init__(self, config: TowardsModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()

        # Feature extractors
        self.structural_extractor = StructuralExtractor.build_from_config()
        self.semantic_extractor = EXTRACTOR.build_from_config()
        self.visual_extractor = VisualExtractor.build_from_config()

        # Define own layers
        self._build_classification_layers(config)

    def forward(self, x: TowardsInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model.
        :return: The output of the model.
        """
        # Feature extractors
        structural_features = self.structural_extractor(x.character_matrix)
        semantic_features = self.semantic_extractor(x.bert)
        visual_features = self.visual_extractor(x.image)

        # Concatenate the inputs
        features = torch.cat(
            (structural_features, semantic_features, visual_features), dim=-1
        )

        # Update the input length of the forward classification layers
        self._update_input_length(features.shape[1])

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
        return cls(TowardsModelConfig(**params))
