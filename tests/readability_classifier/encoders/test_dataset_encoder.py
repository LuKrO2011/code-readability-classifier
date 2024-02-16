from readability_classifier.encoders.dataset_encoder import DatasetEncoder
from tests.readability_classifier.utils.utils import DirTest


class TestDatasetEncoder(DirTest):
    encoder = DatasetEncoder()

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
