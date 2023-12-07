import unittest

from src.readability_classifier.models.encoders.dataset_utils import (
    load_encoded_dataset,
)
from tests.readability_classifier.utils.utils import ENCODED_SCALABRIO_DIR


class TestDatasetUtils(unittest.TestCase):
    def test_load_encoded_dataset(self):
        data_dir = ENCODED_SCALABRIO_DIR.absolute()

        # Load encoded data
        encoded_data = load_encoded_dataset(data_dir)

        # Check if encoded data is not empty
        assert len(encoded_data) > 0
