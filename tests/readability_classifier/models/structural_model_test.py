import pytest

from readability_classifier.models.structural_model import StructuralModel
from readability_classifier.utils.config import StructuralInput
from tests.readability_classifier.models.extractors.structural_extractor_test import (
    create_test_data as create_structural_test_data,
)


@pytest.fixture()
def structural_model():
    return StructuralModel.build_from_config()


def test_forward_pass(structural_model):
    # Create test input data
    structural_input_data = create_test_data()

    # Run the forward pass
    output = structural_model(structural_input_data)

    # Check the output shape
    assert output.shape == (1, 1)


def create_test_data():
    return StructuralInput(create_structural_test_data())
