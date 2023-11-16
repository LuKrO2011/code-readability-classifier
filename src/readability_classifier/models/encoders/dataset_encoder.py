import logging

import torch

from readability_classifier.models.encoders.bert_encoder import BertEncoder
from readability_classifier.models.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)
from readability_classifier.models.encoders.image_encoder import VisualEncoder
from readability_classifier.models.encoders.matrix_encoder import MatrixEncoder


class DatasetEncoder(EncoderInterface):
    """
    A class for encoding code snippets as matrix, with bert and as image.
    Uses the MatrixEncoder, BertEncoder and VisualEncoder for encoding.
    The output is used by the model.
    """

    def __init__(self):
        """
        Initializes the DatasetEncoder.
        """
        self.matrix_encoder = MatrixEncoder()
        self.bert_encoder = BertEncoder()
        self.visual_encoder = VisualEncoder()

    def encode_text(self, code_text: str) -> dict[str, torch.Tensor]:
        """
        Encodes the given code snippet as a matrix, bert and image.
        :param code_text: The code snippet to encode.
        :return: The encoded code snippet.
        """
        matrix = self.matrix_encoder.encode_text(code_text)
        bert = self.bert_encoder.encode_text(code_text)
        image = self.visual_encoder.encode_text(code_text)

        # Log successful encoding
        logging.info("All: Encoding done.")

        return {
            "matrix": matrix["matrix"],
            "input_ids": bert["input_ids"],
            "token_type_ids": bert["token_type_ids"],
            "image": image["image"],
        }

    def encode_dataset(self, unencoded_dataset: list[dict]) -> ReadabilityDataset:
        """
        Encodes the given dataset as matrices, bert and images.
        :param unencoded_dataset: The unencoded dataset.
        :return: The encoded dataset.
        """
        matrix_dataset = self.matrix_encoder.encode_dataset(unencoded_dataset)
        bert_dataset = self.bert_encoder.encode_dataset(unencoded_dataset)
        image_dataset = self.visual_encoder.encode_dataset(unencoded_dataset)

        # Normalize the scores
        scores = [sample["score"] for sample in unencoded_dataset]
        normalized_scores = self._normalize_scores(scores)

        # Combine the datasets
        encoded_dataset = []
        for i in range(len(matrix_dataset)):
            encoded_dataset.append(
                {
                    "matrix": matrix_dataset[i]["matrix"],
                    "input_ids": bert_dataset[i]["input_ids"],
                    "token_type_ids": bert_dataset[i]["token_type_ids"],
                    "image": image_dataset[i]["image"],
                    "score": torch.tensor(normalized_scores[i], dtype=torch.float32),
                }
            )

        # Log the number of samples in the encoded dataset
        logging.info(f"All: Encoding done. Number of samples: {len(encoded_dataset)}")

        return ReadabilityDataset(encoded_dataset)

    @staticmethod
    def _normalize_scores(
        scores: list[float], z_score: bool = True, min_max: bool = True
    ) -> list[float]:
        """
        Normalizes the scores of the dataset using z-score and/or min-max normalization.
        :param scores: The scores to normalize.
        :param z_score: Boolean indicating whether to apply z-score normalization.
        :param min_max: Boolean indicating whether to apply min-max normalization.
        :return: The normalized scores.
        """
        # Divide the scores by the likert max score 5
        scores = [score / 5 for score in scores]

        # Normalized scores to store intermediate results
        normalized_scores = scores.copy()

        # Apply z-score normalization if specified
        if z_score:
            mean = sum(scores) / len(scores)
            std = (sum([(score - mean) ** 2 for score in scores]) / len(scores)) ** 0.5
            normalized_scores = [(score - mean) / std for score in scores]

        # Apply min-max normalization if specified
        if min_max:
            min_score = min(normalized_scores)
            max_score = max(normalized_scores)
            normalized_scores = [
                (score - min_score) / (max_score - min_score)
                for score in normalized_scores
            ]

        return normalized_scores
