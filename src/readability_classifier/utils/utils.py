import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from transformers import BertTokenizer
from yaml import SafeLoader


def read_content_of_file(file: Path, encoding: str = "utf-8") -> str:
    """
    Read the content of a file to str.
    :param file: The given file.
    :param encoding: The given encoding.
    :return: Returns the file content as str.
    :author: Maximilian Jungwirth
    """
    with open(file, encoding=encoding) as file_stream:
        return file_stream.read()


def load_code(file: str) -> str:
    """
    Loads the code from a file.
    :param file: Path to the file.
    :return: Code.
    """
    with open(file) as file:
        return file.read()


def read_java_code_from_file(file_path: str) -> str:
    """
    Reads Java code from a file.
    :param file_path: Path to the file.
    :return: Java code.
    """
    with open(file_path) as file:
        return file.read()


def read_matrix_from_file(file_path: str) -> np.ndarray:
    """
    Reads a matrix from a file.
    :param file_path: Path to the file.
    :return: Matrix.
    """
    # Read the matrix from the file
    data = []
    with open(file_path) as file:
        for line in file:
            values = line.strip().split(",")
            values = [int(val) for val in values if val.strip()]
            if values:
                data.append(values)

    # Create a NumPy array from the data
    return np.array(data)


def save_matrix_to_file(matrix: np.ndarray, file_path: str):
    """
    Saves a matrix to a file.
    :param matrix: Matrix.
    :param file_path: Path to the file.
    """
    # Save the matrix to the file
    with open(file_path, "w") as file:
        for row in matrix:
            row = [str(val) for val in row]
            line = ",".join(row)
            file.write(line + "\n")
