from tempfile import TemporaryDirectory

import pytest

from readability_classifier.models.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.models.encoders.dataset_utils import (
    load_raw_dataset,
    store_encoded_dataset,
)


@pytest.fixture()
def encoder():
    return DatasetEncoder()


@pytest.mark.skip()  # Disabled, because it takes too long
def test_encode_dataset(encoder):
    data_dir = "res/raw_datasets/scalabrio/"

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
