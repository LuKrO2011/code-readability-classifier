import unittest

from src.readability_classifier.encoders.dataset_utils import load_encoded_dataset
from tests.readability_classifier.utils.utils import ENCODED_SCALABRIO_DIR


class TestDatasetUtils(unittest.TestCase):
    def test_load_encoded_dataset(self):
        data_dir = str(ENCODED_SCALABRIO_DIR.absolute())

        # Load encoded data
        encoded_data = load_encoded_dataset(data_dir)

        # Check if encoded data is not empty
        assert len(encoded_data) > 0

    def test_split_dataset(self):
        data_dir = str(ENCODED_SCALABRIO_DIR.absolute())
        encoded_data = load_encoded_dataset(data_dir)
        encoded_data = encoded_data.split(10)
        assert len(encoded_data) == 10
