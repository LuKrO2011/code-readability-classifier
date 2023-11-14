import pytest
import torch

from src.readability_classifier.models.visual_extractor import VisualExtractor

RGB_MIN = 0
RGB_MAX = 255
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3

BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, CHANNELS, IMG_WIDTH, IMG_HEIGHT)


@pytest.fixture()
def visual_extractor():
    return VisualExtractor()


def test_forward_pass(visual_extractor):
    # Create test input data
    input_data = create_test_data()

    # Run the forward pass
    output = visual_extractor(input_data)

    # Check the output shape
    assert output.shape == (1, 6400)


def create_test_data():
    return torch.randint(RGB_MIN, RGB_MAX, SHAPE).float()
