import numpy as np


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
