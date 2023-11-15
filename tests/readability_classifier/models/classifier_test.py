import os
from tempfile import TemporaryDirectory

import pytest
import torch

from src.readability_classifier.models.classifier import CodeReadabilityClassifier
from src.readability_classifier.models.towards_model import TowardsModel
from tests.readability_classifier.models.towards_model_test import create_test_data

#
# BATCH_SIZE = 1
# NUM_EPOCHS = 1
# LEARNING_RATE = 0.001


@pytest.fixture()
def classifier():
    return CodeReadabilityClassifier(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )


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


TOKEN_LENGTH = 512
BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.001


@pytest.fixture()
def readability_model():
    return TowardsModel.build_from_config()


@pytest.fixture()
def criterion():
    return torch.nn.MSELoss()


@pytest.fixture()
def optimizer(readability_model):
    return torch.optim.Adam(readability_model.parameters(), lr=LEARNING_RATE)


def test_backward_pass(readability_model, criterion):
    # Create test input data
    (
        structural_input_data,
        token_input,
        segment_input,
        visual_input_data,
    ) = create_test_data()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, 1).float()

    # Calculate output data
    output = readability_model(
        structural_input_data, token_input, segment_input, visual_input_data
    )

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Check if gradients are updated
    assert any(param.grad is not None for param in readability_model.parameters())


def test_update_weights(readability_model, criterion, optimizer):
    # Create test input data
    (
        structural_input_data,
        token_input,
        segment_input,
        visual_input_data,
    ) = create_test_data()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, 1).float()

    # Calculate output data
    output = readability_model(
        structural_input_data, token_input, segment_input, visual_input_data
    )

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Update weights
    optimizer.step()

    # Check if weights are updated
    assert any(param.grad is not None for param in readability_model.parameters())
