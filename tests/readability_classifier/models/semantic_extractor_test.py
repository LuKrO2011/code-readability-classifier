import pytest
import torch

from src.readability_classifier.models.semantic_extractor import SemanticExtractor

EMBEDDED_MIN = 1
EMBEDDED_MAX = 9999
TOKEN_LENGTH = 512
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, TOKEN_LENGTH)

NUM_CLASSES = 1


@pytest.fixture()
def semantic_extractor():
    return SemanticExtractor()


def test_forward_pass(semantic_extractor):
    # Create test input data
    input_data, token_type_ids, attention_mask, _ = create_test_data()

    # Perform a forward pass
    output = semantic_extractor(input_data, token_type_ids, attention_mask)

    # Check if the output has the expected shape
    assert output.shape == (NUM_CLASSES, BATCH_SIZE)

    # TODO: Check range of output values


def create_test_data():
    # Create test input data
    input_data = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    token_type_ids = torch.zeros(SHAPE).long()
    attention_mask = torch.ones(SHAPE).long()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, NUM_CLASSES).float()

    return input_data, token_type_ids, attention_mask, target_data
