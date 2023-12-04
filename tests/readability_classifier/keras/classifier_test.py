import os

import numpy as np

from readability_classifier.keras.classifier import convert_to_towards_inputs
from readability_classifier.models.encoders.dataset_utils import load_encoded_dataset

RES_DIR = os.path.join(os.path.dirname(__file__), "../../res/")
DATASET_DIR = RES_DIR + "encoded_datasets/all/"


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
