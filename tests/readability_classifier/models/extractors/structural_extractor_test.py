import pytest
import torch

from readability_classifier.models.extractors.structural_extractor import (
    StructuralExtractor,
)

MIN = 1
MAX = 9999
WIDTH = 50
HEIGHT = 305
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, HEIGHT, WIDTH)


@pytest.fixture()
def structural_extractor():
    return StructuralExtractor.build_from_config()


def test_forward_pass(structural_extractor):
    # Create test input data
    input_data = create_test_data()

    # Run the forward pass
    output = structural_extractor(input_data)

    # Check the output shape
    assert output.shape == (1, 4608)


def create_test_data():
    return torch.randint(MIN, MAX, SHAPE).float()
