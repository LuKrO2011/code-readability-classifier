import unittest

import keras

from src.readability_classifier.keras.model import (
    create_classification_model,
    create_towards_model,
)


class TestCreateModel(unittest.TestCase):
    def test_create_classification_model(self):
        model = create_classification_model()
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 4
        assert len(model.layers) == 29
        assert len(model.outputs) == 1

    def test_create_towards_model(self):
        model = create_towards_model()
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 4
        assert len(model.layers) == 29
        assert len(model.outputs) == 1
