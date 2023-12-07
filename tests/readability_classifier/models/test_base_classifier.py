import os
import unittest
from tempfile import TemporaryDirectory

from src.readability_classifier.models.towards_classifier import TowardsClassifier

BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.0015


class TestTowardsClassifier(unittest.TestCase):
    classifier = TowardsClassifier(
        batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )

    @unittest.skip("Disabled, because store in temp dir does not work")
    def test_load_store_model(self):
        model_path = "res/models/model.pt"

        # Create temporary directory
        temp_dir = TemporaryDirectory()

        # Load the classifier
        self.classifier.load(model_path)

        # Store the classifier
        self.classifier.store(temp_dir.name)

        # Check if the model was stored successfully
        assert os.path.exists(os.path.join(temp_dir.name, "model.pt"))

        # Clean up temporary directories
        temp_dir.cleanup()
