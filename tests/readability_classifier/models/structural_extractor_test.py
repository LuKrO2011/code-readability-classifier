import pytest
import torch

from src.readability_classifier.models.structural_extractor import StructuralExtractor

MIN = 1
MAX = 9999
WIDTH = 50
HEIGHT = 350
BATCH_SIZE = 1
CHANNELS = 1
SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture()
def structural_extractor():
    return StructuralExtractor()


def test_forward_pass(structural_extractor):
    # Create test input data
    input_data = create_test_data()

    # Run the forward pass
    output = structural_extractor(input_data)

    # Check the output shape
    assert output.shape == (1, 48384)


def create_test_data():
    return torch.randint(MIN, MAX, SHAPE).float()
