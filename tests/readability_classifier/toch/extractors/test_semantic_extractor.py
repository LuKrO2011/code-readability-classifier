import unittest

import torch

# TODO: Use transformers library instead of local copy
from src.readability_classifier.toch.extractors.semantic_extractor import (
    SemanticExtractor,
)
from src.readability_classifier.utils.config import SemanticInput

EMBEDDED_MIN = 1
EMBEDDED_MAX = 299
TOKEN_LENGTH = 100
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, TOKEN_LENGTH)


def create_test_data():
    input_ids = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    token_type_ids = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    attention_mask = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    segment_ids = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()

    return SemanticInput(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )


class TestSemanticExtractor(unittest.TestCase):
    semantic_extractor = SemanticExtractor.build_from_config()

    @unittest.skip(
        "Torch is not supported anymore. Problems with create_test_data() on GPU."
    )
    def test_forward_pass(self):
        # Create test input data
        semantic_input = create_test_data()

        # Perform a forward pass
        output = self.semantic_extractor(semantic_input)

        # Check if the output has the expected shape
        assert output.shape == (1, 1792)

        # TODO: Check range of output values
