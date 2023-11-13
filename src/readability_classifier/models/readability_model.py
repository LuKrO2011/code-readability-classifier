import torch
from torch import nn as nn

from src.readability_classifier.models.semantic_extractor import SemanticExtractor
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
        self.visual_extractor = VisualExtractor()
        self.semantic_extractor = SemanticExtractor(1)
        self.structural_extractor = StructuralExtractor()

        # Own layers
        self.fc1 = nn.Linear(8064, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        character_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param image: The image tensor.
        :param input_ids:   Tensor of input_ids for the BERT model.
        :param token_type_ids: Tensor of token_type_ids for the BERT model.
        :param attention_mask: Tensor of attention_mask for the BERT model.
        :param character_matrix: The character matrix tensor.
        :return: The output of the model.
        """
        # Feature extractors
        visual_features = self.visual_extractor(image)
        semantic_features = self.semantic_extractor(
            input_ids, token_type_ids, attention_mask
        )
        structural_features = self.structural_extractor(character_matrix)

        # Concatenate the features
        features = torch.cat(
            (visual_features, semantic_features, structural_features), dim=1
        )

        # Apply own layers
        x = nn.functional.relu(self.fc1(features))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x
