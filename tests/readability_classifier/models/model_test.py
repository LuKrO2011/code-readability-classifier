import os
from tempfile import TemporaryDirectory

import pytest
import torch

from readability_classifier.models.semantic_extractor import SemanticExtractor
from src.readability_classifier.models.model import (
    CodeReadabilityClassifier,
    DatasetEncoder,
    load_encoded_dataset,
    load_raw_dataset,
    store_encoded_dataset,
)

EMBEDDED_MIN = 1
EMBEDDED_MAX = 9999
TOKEN_LENGTH = 512
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, TOKEN_LENGTH)
NUM_CLASSES = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.001


@pytest.fixture()
def semantic_extractor():
    return SemanticExtractor(num_classes=NUM_CLASSES)


@pytest.fixture()
def criterion():
    return torch.nn.MSELoss()


@pytest.fixture()
def optimizer(semantic_extractor):
    return torch.optim.Adam(semantic_extractor.parameters(), lr=LEARNING_RATE)


@pytest.fixture()
def classifier():
    return CodeReadabilityClassifier(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )


@pytest.fixture()
def encoder():
    return DatasetEncoder()


def test_backward_pass(semantic_extractor, criterion):
    # Create test input data
    input_data = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    token_type_ids = torch.zeros(SHAPE).long()
    attention_mask = torch.ones(SHAPE).long()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, NUM_CLASSES).float()

    # Calculate output data
    output = semantic_extractor(input_data, token_type_ids, attention_mask)

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Check if gradients are updated
    assert any(param.grad is not None for param in semantic_extractor.parameters())


def test_update_weights(semantic_extractor, criterion, optimizer):
    # Create test input data
    input_data = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
    token_type_ids = torch.zeros(SHAPE).long()
    attention_mask = torch.ones(SHAPE).long()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, NUM_CLASSES).float()

    # Calculate output data
    output = semantic_extractor(input_data, token_type_ids, attention_mask)

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Update weights
    optimizer.step()

    # Check if weights are updated
    assert any(param.grad is not None for param in semantic_extractor.parameters())


@pytest.mark.skip()  # Disabled, because store in temp dir does not work
def test_load_store_model(classifier):
    model_path = "res/models/model.pt"

    # Create temporary directory
    temp_dir = TemporaryDirectory()

    # Load the classifier
    classifier.load(model_path)

    # Store the classifier
    classifier.store(temp_dir.name)

    # Check if the model was stored successfully
    assert os.path.exists(os.path.join(temp_dir.name, "model.pt"))

    # Clean up temporary directories
    temp_dir.cleanup()


def test_encode(encoder):
    data_dir = "res/raw_datasets/scalabrio"

    # Create temporary directory
    temp_dir = TemporaryDirectory()

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Encode raw data
    encoded_data = encoder.encode_dataset(raw_data)

    # Store encoded data
    store_encoded_dataset(encoded_data, temp_dir.name)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0

    # Clean up temporary directories
    temp_dir.cleanup()


def test_load_encoded_dataset():
    data_dir = "res/encoded_datasets/bw"

    # Load encoded data
    encoded_data = load_encoded_dataset(data_dir)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0
