from tempfile import TemporaryDirectory

import numpy as np
import pytest

from readability_classifier.models.encoders.dataset_utils import (
    load_raw_dataset,
    store_encoded_dataset,
)
from readability_classifier.models.encoders.matrix_encoder import (
    MatrixEncoder,
    java_to_structural_representation,
)
from src.readability_classifier.utils.utils import (
    read_java_code_from_file,
    read_matrix_from_file,
    save_matrix_to_file,
)
from tests.readability_classifier.utils.utils import DirTest


@pytest.fixture()
def matrix_encoder():
    return MatrixEncoder()


def test_encode_matrix_dataset(matrix_encoder):
    data_dir = "res/raw_datasets/scalabrio"

    # Create temporary directory
    temp_dir = TemporaryDirectory()

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Encode raw data
    encoded_data = matrix_encoder.encode_dataset(raw_data)

    # Store encoded data
    store_encoded_dataset(encoded_data, temp_dir.name)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0

    # Clean up temporary directories
    temp_dir.cleanup()


def test_encode_matrix_text(matrix_encoder):
    code = """
    // A method for counting
    public void getNumber(){
        int count = 0;
        while(count < 10){
            count++;
        }
    }
    """

    # Encode the code
    encoded_code = matrix_encoder.encode_text(code)

    # Check if encoded code is not empty
    assert len(encoded_code) > 0


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
