import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
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


def get_from_dict(dictionary, key_start: str):
    """
    Get a value from a dict by key_start. The first value of the dict where the key
    starts with key_start is returned.
    :param dictionary: The dict to search in.
    :param key_start: The start of the key.
    :return:
    """
    for key, value in dictionary.items():
        if key.startswith(key_start):
            return value
    raise KeyError(f"Key {key_start} not found in dictionary")


def calculate_precision(tp: int, fp: int) -> float:
    """
    Calculate the precision.
    :param tp: The number of true positives.
    :param fp: The number of false positives.
    :return: The precision.
    """
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def calculate_recall(tp: int, fn: int) -> float:
    """
    Calculate the recall.
    :param tp: The number of true positives.
    :param fn: The number of false negatives.
    :return: The recall.
    """
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate the Matthews correlation coefficient.
    :param tp: The number of true positives.
    :param tn: The number of true negatives.
    :param fp: The number of false positives.
    :param fn: The number of false negatives.
    :return: The Matthews correlation coefficient.
    """
    under_sqrt = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if under_sqrt == 0:
        return 0
    sqrt = math.sqrt(under_sqrt)
    if sqrt == 0:
        return 0
    return (tp * tn - fp * fn) / sqrt


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate the F1 score.
    :param precision: The precision.
    :param recall: The recall.
    :return: The F1 score.
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calculate_auc(precision: float, recall: float) -> float:
    """
    Calculate the area under the curve.
    :param precision: The precision.
    :param recall: The recall.
    :return: The area under the curve.
    """
    return (precision + recall) / 2
