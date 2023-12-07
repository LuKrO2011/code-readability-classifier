import os
import unittest

import keras
import numpy as np

from src.readability_classifier.keras.classifier import (
    Classifier,
    convert_to_towards_inputs,
)
from src.readability_classifier.keras.history_processing import HistoryList
from src.readability_classifier.models.encoders.dataset_utils import (
    load_encoded_dataset,
)

RES_DIR = os.path.join(os.path.dirname(__file__), "../../res/")
DATASET_DIR = RES_DIR + "encoded_datasets/combined/"


def test_convert_to_towards_inputs():
    x = load_encoded_dataset(DATASET_DIR)

    result = convert_to_towards_inputs(x)

    assert len(result) == len(x)
    result_0 = result[0]
    assert isinstance(result_0, dict)
    assert "structure" in result_0
    assert "image" in result_0
    assert "token" in result_0
    assert "segment" in result_0
    assert "label" in result_0
    assert isinstance(result_0["structure"], np.ndarray)
    assert isinstance(result_0["image"], np.ndarray)
    assert isinstance(result_0["token"], np.ndarray)
    assert isinstance(result_0["segment"], np.ndarray)
    assert isinstance(result_0["label"], np.ndarray)
    assert result_0["structure"].shape == (50, 305)
    assert result_0["image"].shape == (128, 128, 3)
    assert result_0["token"].shape == (100,)
    assert result_0["segment"].shape == (100,)
    assert result_0["label"].shape == ()


class TestClassifier(unittest.TestCase):
    def test_train(self):
        # Load the dataset
        x = load_encoded_dataset(DATASET_DIR)

        # Mock the model as a keras.Model class
        class MockModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.history = None

        # Create the classifier
        classifier = Classifier(MockModel(), x)

        # Mock the train_fold method
        def mock_train_fold(*args, **kwargs):
            return {"loss": [1, 2, 3], "val_loss": [4, 5, 6]}

        classifier.train_fold = mock_train_fold

        # Train the model
        k_fold = 2
        history = classifier.train(k_fold=k_fold)

        # Check the history
        assert isinstance(history, HistoryList)
        assert len(history.fold_histories) == 2
        for fold_history in history.fold_histories:
            isinstance(fold_history, dict)
            assert "loss" in fold_history
            assert "val_loss" in fold_history
            assert len(fold_history["loss"]) == 3
            assert len(fold_history["val_loss"]) == 3
