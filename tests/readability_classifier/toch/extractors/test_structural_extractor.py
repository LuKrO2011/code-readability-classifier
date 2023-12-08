import unittest

import torch

from readability_classifier.utils.config import StructuralInput
from src.readability_classifier.toch.extractors.structural_extractor import (
    StructuralExtractor,
)

MIN = 1
MAX = 9999
WIDTH = 50
HEIGHT = 305
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, HEIGHT, WIDTH)


def create_test_data():
    return StructuralInput(torch.randint(MIN, MAX, SHAPE).float())


class TestStructuralExtractor(unittest.TestCase):
    structural_extractor = StructuralExtractor.build_from_config()

    def test_forward_pass(self):
        # Create test input data
        input_data = create_test_data()

        # Run the forward pass
        output = self.structural_extractor(input_data)

        # Check the output shape
        assert output.shape == (1, 4608)
