import os
import unittest

from readability_classifier.models.model import BertEncoder, CodeReadabilityClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(CURRENT_DIR, "res")
MODEL_PATH = os.path.join(RESOURCES_DIR, "res/models/model.pt")
DATA_DIR = (
    "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Dataset/Dataset/"
)


class TestModel(unittest.TestCase):
    """
    Test the model.
    """

    @unittest.skip("Model training takes too long and requires a dataset.")
    def test_train(self) -> None:
        """
        Test the training of the model. After training, the model is stored
        in the current working directory.
        :return: None
        """
        snippets_dir = os.path.join(DATA_DIR, "Snippets")
        csv = os.path.join(DATA_DIR, "scores.csv")

        # Load the data
        data_loader = BertEncoder()
        train_loader, test_loader = data_loader.encode_dataset(csv, snippets_dir)

        # Train and evaluate the model
        classifier = CodeReadabilityClassifier(train_loader, test_loader)
        classifier.train()
        classifier.evaluate()

        # Store the model
        classifier.store("model.pt")

    @unittest.skip("Test requires a model, which is too large for GitHub.")
    def test_predict(self) -> None:
        """
        Test the prediction of the model. Therefore, the model must be loaded
        from the MODEL_PATH.
        :return: None
        """
        # Load the model
        classifier = CodeReadabilityClassifier()
        classifier.load(MODEL_PATH)

        # Predict the readability of a snippet
        snippet = "def foo():\n    return 1"
        readability = classifier.predict(snippet)
        actual_readability = 3.5754477977752686

        # Assert that the predicted readability is the same as the actual
        assert readability == actual_readability
