import pytest

from src.readability_classifier.models.readability_model import ReadabilityModel
from tests.readability_classifier.models.semantic_extractor_test import (
    create_test_data as create_semantic_test_data,
)
from tests.readability_classifier.models.structural_extractor_test import (
    create_test_data as create_structural_test_data,
)
from tests.readability_classifier.models.visual_extractor_test import (
    create_test_data as create_visual_test_data,
)


@pytest.fixture()
def readability_model():
    return ReadabilityModel()


def test_forward_pass(readability_model):
    # Create test input data
    (
        visual_input_data,
        input_data,
        token_type_ids,
        attention_mask,
        structural_input_data,
    ) = create_test_data()

    # Run the forward pass
    output = readability_model(
        visual_input_data,
        input_data,
        token_type_ids,
        attention_mask,
        structural_input_data,
    )

    # Check the output shape
    assert output.shape == (1, 1)


def create_test_data():
    # Create test input data
    input_data, token_type_ids, attention_mask, _ = create_semantic_test_data()
    structural_input_data, _ = create_structural_test_data()
    visual_input_data, _ = create_visual_test_data()

    return (
        visual_input_data,
        input_data,
        token_type_ids,
        attention_mask,
        structural_input_data,
    )