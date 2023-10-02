import pytest
import torch

from src.readability_classifier.models.model import CNNModel

EMBEDDED_MIN = 1
EMBEDDED_MAX = 9999
TOKEN_LENGTH = 512
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, TOKEN_LENGTH)
NUM_CLASSES = 1


@pytest.fixture()
def model():
    return CNNModel(num_classes=NUM_CLASSES)


def test_forward_pass(model):
    # Create test x_batch with shape (1, 512) and values between 1 and 9999
    input_data = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()

    # Create test attention mask with shape (1, 512)
    attention_mask = torch.ones(SHAPE).long()

    # Perform a forward pass
    output = model(input_data, attention_mask)

    # Check if the output has the expected shape
    assert output.shape == (NUM_CLASSES, model.get_num_classes())


#
# def test_backward_pass(model):
#     # Create input data and target data (for backpropagation)
#     input_data = torch.randint(EMBEDDED_MIN, EMBEDDED_MAX, SHAPE).long()
#     target_data = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).long()
#
#     # Perform a backward pass
#     loss = model.criterion(outputs, y_batch)
#     loss.backward()
#
#     # Check if gradients are updated
#     assert any(param.grad is not None for param in model.parameters())
#
#
# def test_prediction(model):
#     pass
