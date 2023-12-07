import unittest

import torch

# TODO: Use transformers library instead of local copy
from src.readability_classifier.models.extractors.semantic_extractor import (
    SemanticExtractor,
)

EMBEDDED_MIN = 1
EMBEDDED_MAX = 299
TOKEN_LENGTH = 100
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, TOKEN_LENGTH)


def create_test_data():
    token_input = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    segment_input = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    return token_input, segment_input


class TestSemanticExtractor(unittest.TestCase):
    semantic_extractor = SemanticExtractor.build_from_config()

    def test_forward_pass(self):
        # Create test input data
        token_input, segment_input = create_test_data()

        # Perform a forward pass
        output = self.semantic_extractor(token_input, segment_input)

        # Check if the output has the expected shape
        assert output.shape == (1, 1792)

        # TODO: Check range of output values
