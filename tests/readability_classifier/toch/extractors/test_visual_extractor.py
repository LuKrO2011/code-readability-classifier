import unittest

import torch

from src.readability_classifier.utils.config import VisualInput
from src.readability_classifier.toch.extractors.visual_extractor import VisualExtractor

RGB_MIN = 0
RGB_MAX = 255
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3

BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, CHANNELS, IMG_WIDTH, IMG_HEIGHT)


def create_test_data():
    return VisualInput(torch.randint(RGB_MIN, RGB_MAX, SHAPE).float())


class TestVisualExtractor(unittest.TestCase):
    visual_extractor = VisualExtractor.build_from_config()

    def test_forward_pass(self):
        # Create test input data
        input_data = create_test_data()

        # Run the forward pass
        output = self.visual_extractor(input_data)

        # Check the output shape
        assert output.shape == (1, 12544)
