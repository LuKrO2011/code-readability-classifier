import pytest

from readability_classifier.utils.config import TowardsInput
from src.readability_classifier.models.towards_model import TowardsModel
from tests.readability_classifier.models.extractors.semantic_extractor_test import (
    create_test_data as create_semantic_test_data,
)
from tests.readability_classifier.models.extractors.structural_extractor_test import (
    create_test_data as create_structural_test_data,
)
from tests.readability_classifier.models.extractors.visual_extractor_test import (
    create_test_data as create_visual_test_data,
)


@pytest.fixture()
def readability_model():
    return TowardsModel.build_from_config()


def test_forward_pass(readability_model):
    # Create test input data
    input_data = create_test_data()

    # Run the forward pass
    output = readability_model(input_data)

    # Check the output shape
    assert output.shape == (1, 1)


def create_test_data():
    structural_input_data = create_structural_test_data()
    token_input, segment_input = create_semantic_test_data()
    visual_input_data = create_visual_test_data()

    return TowardsInput(
        structural_input_data,
        token_input,
        segment_input,
        visual_input_data,
    )
