import unittest

from readability_classifier.encoders.dataset_utils import load_encoded_dataset
from src.readability_classifier.keas.keras_model_runner import KerasModelRunner
from tests.readability_classifier.utils.utils import ENCODED_BW_DIR, DirTest


class TestKerasModelRunner(DirTest):
    def setUp(self):
        super().setUp()
        self.encoded_data = load_encoded_dataset(ENCODED_BW_DIR)

    @unittest.skip("Takes to long.")
    def test_run_with_cross_validation(self):
        # Mock the parsed arguments
        class MockParsedArgs:
            def __init__(self, save: str = self.output_dir):
                self.save = save

        # Run the model runner
        model_runner = KerasModelRunner()
        model_runner._run_with_cross_validation(
            parsed_args=MockParsedArgs(), encoded_data=self.encoded_data
        )
