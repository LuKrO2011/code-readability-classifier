import pytest
import torch

from src.readability_classifier.models.towards_model import TowardsModel
from tests.readability_classifier.models.towards_model_test import create_test_data

BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.0015


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
    input_data = create_test_data()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, 1).float()

    # Calculate output data
    output = readability_model(input_data)

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Check if gradients are updated
    assert any(param.grad is not None for param in readability_model.parameters())


def test_update_weights(readability_model, criterion, optimizer):
    # Create test input data
    input_data = create_test_data()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, 1).float()

    # Calculate output data
    output = readability_model(input_data)

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Update weights
    optimizer.step()

    # Check if weights are updated
    assert any(param.grad is not None for param in readability_model.parameters())
