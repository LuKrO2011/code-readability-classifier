import numpy as np

from readability_classifier.encoders.matrix_encoder import (
    MatrixEncoder,
    java_to_structural_representation,
)
from src.readability_classifier.utils.utils import (
    read_java_code_from_file,
    read_matrix_from_file,
    save_matrix_to_file,
)
from tests.readability_classifier.utils.utils import (
    MI_RAW_DIR,
    MI_STRUCTURAL_DIR,
    DirTest,
)


class TestMatrixEncoder(DirTest):
    matrix_encoder = MatrixEncoder()

    def test_encode_matrix_text(self):
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
        encoded_code = self.matrix_encoder.encode_text(code)

        # Check if encoded code is not empty
        assert len(encoded_code) > 0


class TestStructural(DirTest):
    def template_structural(self, name):
        java_name = name + ".java"
        java_path = str(MI_RAW_DIR) + "/" + java_name
        matrix_name = name + ".java.matrix"
        matrix_path = str(MI_STRUCTURAL_DIR) + "/" + matrix_name

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
