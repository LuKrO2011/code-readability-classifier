import logging

import numpy as np
import torch

from readability_classifier.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)


class MatrixEncoder(EncoderInterface):
    """
    A class for encoding code snippets as character matrices (ASCII values).
    """

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
