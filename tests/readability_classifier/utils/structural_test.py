import numpy as np

from readability_classifier.models.encoders.matrix_encoder import (
    java_to_structural_representation,
)
from readability_classifier.utils.utils import (
    read_java_code_from_file,
    read_matrix_from_file,
    save_matrix_to_file,
)
from tests.readability_classifier.utils.utils import DirTest


class TestStructural(DirTest):
    java_dir = "res/mi/raw/"
    matrix_dir = "res/mi/structural/"

    def template_structural(self, name):
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

    def test_Buse14(self):
        self.template_structural("Buse14")

    def test_Buse23(self):
        self.template_structural("Buse23")
