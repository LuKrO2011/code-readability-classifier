import torch
from torch import nn as nn

from src.readability_classifier.models.semantic_extractor import (
    BertConfig,
    SemanticExtractor,
)
from src.readability_classifier.models.structural_extractor import StructuralExtractor
from src.readability_classifier.models.visual_extractor import VisualExtractor


class ReadabilityModel(nn.Module):
    """
    A code readability classifier based on a CNN model.
    The model consists of a visual, a semantic and a structural feature extractor plus
    own layers.
    The input consists of an image, a bert encoded code snippet and a
    character matrix. The image is of size (3, 128, 128), the bert encoded code snippet
    is of size (1, 512) and the character matrix is of size (350, 50). The output is a
    single value representing the readability of the code snippet.

    The own layers consist of:
    1. Fully connected layer
    2. Dropout layer
    3. Fully connected layer
    4. Fully connected layer
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        # Feature extractors
        self.structural_extractor = StructuralExtractor()
        self.bert_config = BertConfig()
        self.semantic_extractor = SemanticExtractor(self.bert_config)
        self.visual_extractor = VisualExtractor()

        # TODO: Get from feature extractors?
        # Specify input size
        self.structural_features_size = 41472
        self.semantic_features_size = 10560
        self.visual_features_size = 6400
        self.concatenated_size = (
            self.structural_features_size
            + self.semantic_features_size
            + self.visual_features_size
        )

        # Define own layers
        self.dense1 = nn.Linear(self.concatenated_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, 16)
        self.random_detail = nn.Linear(16, 1)
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
