from pathlib import Path

import torch

from readability_classifier.models.base_model import BaseModel
from readability_classifier.models.extractors.semantic_extractor import (
    SemanticExtractor,
)
from readability_classifier.models.extractors.semantic_extractor_krod import (
    KrodSemanticExtractor,
)
from readability_classifier.utils.config import BaseModelConfig, SemanticInput

USE_KROD = True


class SemanticModelConfig(BaseModelConfig):
    """
    The config for the SemanticModel.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the config.
        """
        super().__init__(**kwargs)
        if USE_KROD:
            self.input_length = kwargs.get("input_length", 640)
        else:
            self.input_length = kwargs.get("input_length", 1792)
        self.output_length = kwargs.get("output_length", 1)
        self.dropout = kwargs.get("dropout", 0.5)


class SemanticModel(BaseModel):
    """
    A code readability model based on the semantic features of the code.
    The model consists of a semantic feature extractor plus own layers.
    The input consists of a token and a segment embedding for bert.
    The output is a single value representing the readability of the code snippet.
    """

    def __init__(self, config: SemanticModelConfig) -> None:
        """
        Initialize the model.
        :param config: The config for the model.
        """
        super().__init__()

        # Feature extractors
        if USE_KROD:
            self.semantic_extractor = KrodSemanticExtractor.build_from_config()
        else:
            self.semantic_extractor = SemanticExtractor.build_from_config()

        # Define own layers
        self._build_classification_layers(config)

    def forward(self, x: SemanticInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model containing the bert embeddings.
        :return: The output of the model.
        """
        # Feature extractors
        semantic_features = self.semantic_extractor(x.token_input, x.segment_input)

        # Pass through dense layers
        return self._forward_classification_layers(semantic_features)

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(SemanticModelConfig(**params))
