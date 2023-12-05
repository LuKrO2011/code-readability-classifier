from tempfile import TemporaryDirectory

import pytest

from readability_classifier.models.encoders.bert_encoder import BertEncoder
from readability_classifier.models.encoders.dataset_utils import (
    load_raw_dataset,
    store_encoded_dataset,
)
from tests.readability_classifier.utils.utils import RAW_SCALABRIO_DIR


@pytest.fixture()
def bert_encoder():
    return BertEncoder()


def test_encode_bert(bert_encoder):
    data_dir = RAW_SCALABRIO_DIR.absolute()

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


def test_encode_text(bert_encoder):
    data_dir = RAW_SCALABRIO_DIR.absolute()

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Get the first code snippet
    text = raw_data[0]["code_snippet"]

    # Encode text
    encoded_text = bert_encoder.encode_text(text)

    # Check if encoded text is not empty
    assert len(encoded_text) > 0
