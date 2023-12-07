import unittest

from src.readability_classifier.models.structural_model import StructuralModel
from src.readability_classifier.utils.config import StructuralInput
from tests.readability_classifier.models.extractors.test_structural_extractor import (
    create_test_data as create_structural_test_data,
)


def create_test_data():
    return StructuralInput(create_structural_test_data())


class TestStructuralModel(unittest.TestCase):
    structural_model = StructuralModel.build_from_config()

    def test_forward_pass(self):
        # Create test input data
        structural_input_data = create_test_data()

        # Run the forward pass
        output = self.structural_model(structural_input_data)

        # Check the output shape
        assert output.shape == (1, 1)
