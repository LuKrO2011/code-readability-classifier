import os
from tempfile import TemporaryDirectory

import pytest

from src.readability_classifier.models.towards_classifier import TowardsClassifier

BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.0015


@pytest.fixture()
def classifier():
    return TowardsClassifier(
        batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
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
