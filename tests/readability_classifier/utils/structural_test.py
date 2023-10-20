import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from readability_classifier.utils.strucutral import (
    java_to_structural_representation,
    read_java_code_from_file,
    read_matrix_from_file,
    save_matrix_to_file,
)


def test_buse14():
    java_path = "res/mi/raw/Buse14.java"
    matrix_path = "res/mi/structural/Buse14.java.matrix"
    java_code = read_java_code_from_file(java_path)
    matrix_actual = java_to_structural_representation(java_code)
    matrix_expected = read_matrix_from_file(matrix_path)
    assert np.array_equal(matrix_actual, matrix_expected)


def test_buse23():
    java_path = "res/mi/raw/Buse23.java"
    matrix_path = "res/mi/structural/Buse23.java.matrix"
    java_code = read_java_code_from_file(java_path)
    matrix_actual = java_to_structural_representation(java_code)
    matrix_expected = read_matrix_from_file(matrix_path)
    assert np.array_equal(matrix_actual, matrix_expected)


# TODO: Refactor this and dataset_converter_test.py
class TestStructural(unittest.TestCase):
    output_dir = None  # Set to "output" to generate output
    java_dir = "res/mi/raw/"
    matrix_dir = "res/mi/structural/"

    def setUp(self):
        # Create temporary directories for testing if output directory is None
        if self.output_dir is None:
            self.temp_dir = TemporaryDirectory()
            self.output_dir = self.temp_dir.name
        else:
            self.temp_dir = None

    def tearDown(self):
        # Clean up temporary directories
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

    def test_buse14(self):
        java_path = self.java_dir + "Buse14.java"
        matrix_path = self.matrix_dir + "Buse14.java.matrix"

        # Convert java to matrix
        java_code = read_java_code_from_file(java_path)
        matrix_actual = java_to_structural_representation(java_code)

        # Save matrix to file in output dir
        save_matrix_to_file(matrix_actual, self.output_dir + "/Buse14.java.matrix")

        # Load expected and actual matrix
        matrix_actual = read_matrix_from_file(self.output_dir + "/Buse14.java.matrix")
        matrix_expected = read_matrix_from_file(matrix_path)

        # Check if matrices are equal
        assert np.array_equal(matrix_actual, matrix_expected)

    @pytest.mark.parametrize("name", ["Buse14", "Buse23"])
    def test_structural(self, name):
        java_name = name + ".java"
        java_path = self.java_dir + java_name
        matrix_name = name + ".java.matrix"
        matrix_path = self.matrix_dir + matrix_name

        # Convert java to matrix
        java_code = read_java_code_from_file(java_path)
        matrix_actual = java_to_structural_representation(java_code)

        # Save matrix to file in output dir
        save_matrix_to_file(matrix_actual, self.output_dir + "/" + matrix_name)

        # Load expected and actual matrix
        matrix_actual = read_matrix_from_file(self.output_dir + "/" + matrix_name)
        matrix_expected = read_matrix_from_file(matrix_path)

        # Check if matrices are equal
        assert np.array_equal(matrix_actual, matrix_expected)
