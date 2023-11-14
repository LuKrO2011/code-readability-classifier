import os
from tempfile import TemporaryDirectory

import pytest
import torch

from readability_classifier.models.encoders.bert_encoder import BertEncoder
from readability_classifier.models.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.models.encoders.dataset_utils import (
    load_encoded_dataset,
    load_raw_dataset,
    store_encoded_dataset,
)
from readability_classifier.models.encoders.image_encoder import VisualEncoder
from readability_classifier.models.encoders.matrix_encoder import MatrixEncoder
from src.readability_classifier.models.classifier import CodeReadabilityClassifier
from src.readability_classifier.models.model import ReadabilityModel
from tests.readability_classifier.models.readability_model_test import create_test_data

EMBEDDED_MIN = 1
EMBEDDED_MAX = 9999
TOKEN_LENGTH = 512
BATCH_SIZE = 1
SHAPE = (BATCH_SIZE, TOKEN_LENGTH)
NUM_EPOCHS = 1
LEARNING_RATE = 0.001


@pytest.fixture()
def readability_model():
    return ReadabilityModel()


@pytest.fixture()
def criterion():
    return torch.nn.MSELoss()


@pytest.fixture()
def optimizer(readability_model):
    return torch.optim.Adam(readability_model.parameters(), lr=LEARNING_RATE)


@pytest.fixture()
def classifier():
    return CodeReadabilityClassifier(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )


@pytest.fixture()
def bert_encoder():
    return BertEncoder()


@pytest.fixture()
def visual_encoder():
    return VisualEncoder()


@pytest.fixture()
def matrix_encoder():
    return MatrixEncoder()


@pytest.fixture()
def encoder():
    return DatasetEncoder()


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


def test_encode_bert(bert_encoder):
    data_dir = "res/raw_datasets/scalabrio"

    # Create temporary directory
    temp_dir = TemporaryDirectory()

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Encode raw data
    encoded_data = bert_encoder.encode_dataset(raw_data)

    # Store encoded data
    store_encoded_dataset(encoded_data, temp_dir.name)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0

    # Clean up temporary directories
    temp_dir.cleanup()


@pytest.mark.skip()  # Disabled, because it takes too long
def test_encode_visual_dataset(visual_encoder):
    data_dir = "res/raw_datasets/scalabrio"

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Encode raw data
    encoded_data = visual_encoder.encode_dataset(raw_data)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0

    # # Show the first image
    # bytes_to_image(tensor_to_bytes(encoded_data[0]["image"]), "code.png")
    # from PIL import Image
    #
    # img = Image.open("code.png")
    # img.show()


def test_encode_visual_text(visual_encoder):
    # Sample Java code
    code = """
    // A method for counting
    public void getNumber(){
        int count = 0;
        while(count < 10){
            count++;
        }
    }
    """

    # Encode the code
    encoded_code = visual_encoder.encode_text(code)

    # Check if encoded code is not empty
    assert len(encoded_code) > 0

    # Show the image
    # bytes_to_image(encoded_code, "code.png")
    # from PIL import Image
    # img = Image.open("code.png")
    # img.show()


def test_encode_matrix_dataset(matrix_encoder):
    data_dir = "res/raw_datasets/scalabrio"

    # Create temporary directory
    temp_dir = TemporaryDirectory()

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Encode raw data
    encoded_data = matrix_encoder.encode_dataset(raw_data)

    # Store encoded data
    store_encoded_dataset(encoded_data, temp_dir.name)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0

    # Clean up temporary directories
    temp_dir.cleanup()


def test_encode_matrix_text(matrix_encoder):
    code = """
    // A method for counting
    public void getNumber(){
        int count = 0;
        while(count < 10){
            count++;
        }
    }
    """

    # Encode the code
    encoded_code = matrix_encoder.encode_text(code)

    # Check if encoded code is not empty
    assert len(encoded_code) > 0


@pytest.mark.skip()  # Disabled, because it takes too long
def test_encode_dataset(encoder):
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


def test_encode_text(encoder):
    code = """
    // A method for counting
    public void getNumber(){
        int count = 0;
        while(count < 10){
            count++;
        }
    }
    """

    # Encode the code
    encoded_code = encoder.encode_text(code)

    # Check if encoded code is not empty
    assert len(encoded_code) > 0


def test_load_encoded_dataset():
    data_dir = "res/encoded_datasets/bw"

    # Load encoded data
    encoded_data = load_encoded_dataset(data_dir)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0
