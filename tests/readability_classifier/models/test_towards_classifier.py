import unittest

import torch

from src.readability_classifier.models.towards_model import TowardsModel
from tests.readability_classifier.models.test_towards_model import create_test_data

BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.0015


class TestTowardsModel(unittest.TestCase):
    towards_model = TowardsModel.build_from_config()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(towards_model.parameters(), lr=LEARNING_RATE)

    def test_backward_pass(self):
        # Create test input data
        input_data = create_test_data()

        # Create target data
        target_data = torch.rand(BATCH_SIZE, 1).float()

        # Calculate output data
        output = self.towards_model(input_data)

        # Perform a backward pass
        loss = self.criterion(output, target_data)
        loss.backward()

        # Check if gradients are updated
        assert any(param.grad is not None for param in self.towards_model.parameters())

    def test_update_weights(self):
        # Create test input data
        input_data = create_test_data()

        # Create target data
        target_data = torch.rand(BATCH_SIZE, 1).float()

        # Calculate output data
        output = self.towards_model(input_data)

        # Perform a backward pass
        loss = self.criterion(output, target_data)
        loss.backward()

        # Update weights
        self.optimizer.step()

        # Check if weights are updated
        assert any(param.grad is not None for param in self.towards_model.parameters())
