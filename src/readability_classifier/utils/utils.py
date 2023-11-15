import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
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


def store_as_txt(stratas: list[list[str]], output_dir: str) -> None:
    """
    Store the sampled Java code snippet paths in a txt file.
    :param stratas: The sampled Java code snippet paths
    :param output_dir: The directory where the txt file should be stored
    :return: None
    """
    with open(os.path.join(output_dir, "stratas.txt"), "w") as file:
        for idx, stratum in enumerate(stratas):
            file.write(f"Stratum {idx}:\n")
            for snippet in stratum:
                file.write(f"{snippet}\n")


def list_java_files(directory: str) -> list[str]:
    """
    List all Java files in a directory.
    :param directory: The directory to search for Java files
    :return: A list of Java files
    """
    java_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.abspath(os.path.join(root, file)))

    return java_files


def load_code(file: str) -> str:
    """
    Loads the code from a file.
    :param file: Path to the file.
    :return: Code.
    """
    with open(file) as file:
        return file.read()


def image_to_bytes(image_path: str) -> bytes:
    """
    Converts an image to bytes.
    :param image_path: The path to the image
    :return: The image as bytes
    """
    with open(image_path, "rb") as f:
        return f.read()


def bytes_to_image(image: bytes, image_path: str) -> None:
    """
    Converts bytes to an image.
    :param image: The image as bytes
    :param image_path: The path where the image should be stored
    :return: None
    """
    with open(image_path, "wb") as f:
        f.write(image)


def copy_files(from_dir: str, to_dir: str) -> None:
    """
    Copies all files from directory.
    :param from_dir: The directory to copy from.
    :param to_dir: The directory to copy to.
    :return: None
    """
    for file in os.listdir(from_dir):
        from_file = os.path.join(from_dir, file)
        to_file = os.path.join(to_dir, file)
        if os.path.isfile(from_file):
            shutil.copy2(from_file, to_file)


# TODO: Remove blur (!= 0 or 255) from image
def open_image_as_tensor(image_path: str) -> Tensor:
    """
    Opens a png image as rgb tensor. Removes the alpha channel and transforms the values
    to float32. The shape of the tensor is (3, height, width).
    :param image_path: The path to the image
    :return: The image as a tensor
    """
    # Open the image using PIL
    img = Image.open(image_path)

    # Convert PIL image to NumPy array
    img_array = np.array(img)

    # Remove the alpha channel
    img_array = img_array[:, :, :3]

    # Transpose the array to get the shape (3, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Convert NumPy array to tensor
    return torch.tensor(img_array, dtype=torch.float32)


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


def load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Loads a yaml file to a dict.
    :param path: The path to the yaml file.
    :return: Returns the loaded yaml as dict.
    """
    # Read file
    try:
        raw_str = read_content_of_file(path)
    except FileNotFoundError:
        logging.warning(f"Yaml file {path} not found.")
        return {}

    # Parse yaml
    try:
        dic = yaml.load(raw_str, Loader=SafeLoader)
    except yaml.YAMLError as e:
        logging.warning(f"Yaml file {path} could not be parsed.")
        logging.warning(e)
        return {}

    # Return dict
    if dic is None:
        return {}
    return dic


def save_content_to_file(content: str, file: Path) -> None:
    """
    Saves the given content to the specified file.
    :param file: The given file.
    :param content: The given content.
    :return: None
    """
    with open(file, "w", encoding="utf-8") as file_stream:
        file_stream.write(content)
