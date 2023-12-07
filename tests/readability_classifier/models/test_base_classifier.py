import os
import unittest

from src.readability_classifier.models.towards_classifier import TowardsClassifier
from tests.readability_classifier.utils.utils import DirTest

BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.0015


class TestTowardsClassifier(DirTest):
    classifier = TowardsClassifier(
        batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )

    @unittest.skip("Only works if cuda is available")
    def test_load_store_model(self):
        model_path = "res/models/model.pt"

        # Load the classifier
        self.classifier.load(model_path)

        # Store the classifier
        self.classifier.store(self.output_dir)

        # Check if the model was stored successfully
        assert os.path.exists(os.path.join(self.output_dir, "model.pt"))
