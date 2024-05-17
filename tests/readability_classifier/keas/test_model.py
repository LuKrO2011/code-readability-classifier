import unittest

import keras
from keras import KerasTensor, layers, models

from src.readability_classifier.keas.legacy_encoders import MAX_LEN
from src.readability_classifier.keas.model import (
    BertConfig,
    BertEmbedding,
    create_classification_layers,
    create_semantic_extractor,
    create_semantic_model,
    create_structural_extractor,
    create_structural_model,
    create_towards_model,
    create_visual_extractor,
    create_visual_model,
)

CONCATENATED_LENGTH = 21056


def print_model_stats(model):
    print(len(model.inputs))
    print(len(model.layers))
    print(len(model.outputs))


def print_tensor_stats(list_of_tensors: list):
    for tensor in list_of_tensors:
        print(tensor.shape)


class TestCreateModel(unittest.TestCase):
    def test_create_classification_model(self):
        concatenated = keras.Input(shape=(CONCATENATED_LENGTH,), name="input")
        classification_layers = create_classification_layers(concatenated)
        assert isinstance(classification_layers, KerasTensor)
        assert classification_layers.shape == (None, 1)

    def test_create_structural_extractor(self):
        input_shape = (None, 50, 305)
        output_shape = (None, 4608)

        model_input, flattened = create_structural_extractor()

        assert isinstance(model_input, KerasTensor)
        assert model_input.shape == input_shape
        assert isinstance(flattened, KerasTensor)
        assert flattened.shape == output_shape

    def test_create_structural_model(self):
        model = create_structural_model()
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 1
        assert len(model.layers) == 13
        assert len(model.outputs) == 1

    def test_create_semantic_extractor(self):
        input_shape = (None, MAX_LEN)
        output_shape = (None, 64)

        token_input, segment_input, gru = create_semantic_extractor()

        assert isinstance(token_input, KerasTensor)
        assert token_input.shape == input_shape
        assert isinstance(segment_input, KerasTensor)
        assert segment_input.shape == input_shape
        assert isinstance(gru, KerasTensor)
        assert gru.shape == output_shape

    def test_create_semantic_model(self):
        model = create_semantic_model()
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2
        assert len(model.layers) == 11
        assert len(model.outputs) == 1

    def test_create_visual_extractor(self):
        input_shape = (None, 128, 128, 3)
        output_shape = (None, 16384)

        model_input, flattened = create_visual_extractor()

        assert isinstance(model_input, KerasTensor)
        assert model_input.shape == input_shape
        assert isinstance(flattened, KerasTensor)
        assert flattened.shape == output_shape

    def test_create_visual_model(self):
        model = create_visual_model()
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 1
        assert len(model.layers) == 12
        assert len(model.outputs) == 1

    def test_create_towards_model(self):
        model = create_towards_model()
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 4
        assert len(model.layers) == 29
        assert len(model.outputs) == 1

    def test_bert_embedding(self):
        # Craft a bert embedding model
        input_shape = (MAX_LEN,)
        token_input = layers.Input(shape=input_shape, name="token")
        segment_input = layers.Input(shape=input_shape, name="segment")
        bert_layer = BertEmbedding(config=BertConfig(max_sequence_length=MAX_LEN))(
            [token_input, segment_input]
        )
        bert_layer_output = bert_layer[0]
        model = models.Model(
            inputs=[token_input, segment_input], outputs=bert_layer_output
        )

        # Check the model
        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2
        assert len(model.layers) == 4
        assert len(model.outputs) == 1
