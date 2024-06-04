import logging

import numpy as np
import torch

from src.readability_classifier.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)


class MatrixEncoder(EncoderInterface):
    """
    A class for encoding code snippets as character matrices (ASCII values).
    """

    def encode_dataset(self, unencoded_dataset: list[dict]) -> ReadabilityDataset:
        """
        Encodes the given dataset as matrices.
        :param unencoded_dataset: The unencoded dataset.
        :return: The encoded dataset.
        """
        encoded_dataset = []

        # Log the number of code snippets to encode
        logging.info(
            f"Matrix: Number of code snippets to encode: {len(unencoded_dataset)}"
        )

        # Encode the code snippets
        for sample in unencoded_dataset:
            encoded_dataset.append(
                {
                    "matrix": torch.tensor(
                        java_to_structural_representation(sample["code_snippet"]),
                        dtype=torch.float32,  # Why not int?
                    ),
                }
            )

        # Log the number of samples in the encoded dataset
        logging.info(
            f"Matrix: Encoding done. Number of samples: {len(encoded_dataset)}"
        )

        return ReadabilityDataset(encoded_dataset)

    def encode_text(self, text: str) -> dict:
        """
        Encodes the given text as a matrix.
        :param text: The text to encode.
        :return: The encoded text as a matrix.
        """
        # Encode the code snippet
        matrix = java_to_structural_representation(text)

        # Log successful encoding
        logging.info("Matrix: Encoding done.")

        return {"matrix": torch.tensor(matrix, dtype=torch.float32)}  # Why not int?


def java_to_structural_representation(
    java_code: str, max_rows: int = 50, max_cols: int = 305
) -> np.ndarray:
    """
    Converts Java code to structural representation.
    :param java_code: Java code.
    :param max_rows: Maximum number of rows.
    :param max_cols: Maximum number of columns.
    :return: Structural representation.
    """
    # Initialize an empty 2D character matrix with values -1
    character_matrix = np.full((max_rows, max_cols), -1, dtype=np.int32)

    # Convert Java code to ASCII values and populate the character matrix
    lines = java_code.splitlines(keepends=True)
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            if row < max_rows and col < max_cols:
                character_matrix[row, col] = ord(char)

    return character_matrix
