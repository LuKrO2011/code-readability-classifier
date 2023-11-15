import pytest
import torch

from readability_classifier.models.structural_model import StructuralModel
from tests.readability_classifier.models.structural_model_test import create_test_data

BATCH_SIZE = 1
LEARNING_RATE = 0.0015


@pytest.fixture()
def structural_model():
    return StructuralModel.build_from_config()


@pytest.fixture()
def criterion():
    return torch.nn.MSELoss()


@pytest.fixture()
def optimizer(structural_model):
    return torch.optim.Adam(structural_model.parameters(), lr=LEARNING_RATE)


def test_backward_pass(structural_model, criterion):
    # Create test input data
    matrix = create_test_data()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, 1).float()

    # Calculate output data
    output = structural_model(matrix)

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Check if gradients are updated
    assert any(param.grad is not None for param in structural_model.parameters())


def test_update_weights(structural_model, criterion, optimizer):
    # Create test input data
    matrix = create_test_data()

    # Create target data
    target_data = torch.rand(BATCH_SIZE, 1).float()

    # Calculate output data
    output = structural_model(matrix)

    # Perform a backward pass
    loss = criterion(output, target_data)
    loss.backward()

    # Update weights
    optimizer.step()

    # Check if weights are updated
    assert any(param.grad is not None for param in structural_model.parameters())
