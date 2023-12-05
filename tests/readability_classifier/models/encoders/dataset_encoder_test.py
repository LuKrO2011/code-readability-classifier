import pytest

from readability_classifier.models.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.models.encoders.dataset_utils import (
    load_raw_dataset,
    store_encoded_dataset,
)
from tests.readability_classifier.utils.utils import RAW_COMBINED_DIR, DirTest


class TestDatasetEncoder(DirTest):
    encoder = DatasetEncoder()

    @pytest.mark.skip()  # Disabled, because it takes too long
    def test_encode_dataset(self):
        data_dir = RAW_COMBINED_DIR.absolute()

        # Load raw data
        raw_data = load_raw_dataset(data_dir)

        # Encode raw data
        encoded_data = self.encoder.encode_dataset(raw_data)

        # Store encoded data
        store_encoded_dataset(encoded_data, self.output_dir)

        # Check if encoded data is not empty
        assert len(encoded_data) > 0

    def test_encode_text(self):
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
        encoded_code = self.encoder.encode_text(code)

        # Check if encoded code is not empty
        assert len(encoded_code) > 0
