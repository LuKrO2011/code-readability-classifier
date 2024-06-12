import unittest

from src.readability_classifier.encoders.dataset_encoder import DatasetEncoder
from src.readability_classifier.encoders.dataset_utils import load_encoded_dataset
from src.readability_classifier.keas.model_runner import KerasModelRunner
from src.readability_classifier.utils.utils import read_content_of_file
from tests.readability_classifier.utils.utils import (
    ENCODED_BW_DIR,
    ENCODED_COMBINED_DIR,
    TOWARDS_CODE_SNIPPET,
    TOWARDS_MODEL,
    DirTest,
)


class TestKerasModelRunner(DirTest):
    def setUp(self):
        super().setUp()
        self.encoded_data = load_encoded_dataset(str(ENCODED_BW_DIR))

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

    @unittest.skip("Takes to long.")
    def test_run_with_cross_validation_finetune(self):
        # Mock the parsed arguments
        class MockParsedArgs:
            def __init__(self, save: str = self.output_dir):
                self.save = save
                self.fine_tune = TOWARDS_MODEL
                self.batch_size = 42
                self.epochs = 3
                self.learning_rate = 0.0015

        # Run the model runner
        model_runner = KerasModelRunner()
        model_runner._run_with_cross_validation(
            parsed_args=MockParsedArgs(), encoded_data=self.encoded_data
        )

    def test_run_predict(self):
        # Mock the parsed arguments
        class MockParsedArgs:
            def __init__(self):
                self.model = str(TOWARDS_MODEL)

        # Get the encoded data
        code = read_content_of_file(TOWARDS_CODE_SNIPPET)
        dataset = [{'name': TOWARDS_CODE_SNIPPET, 'code_snippet': code}]
        encoded_dataset = DatasetEncoder().encode_dataset(dataset)

        # Run the model runner
        model_runner = KerasModelRunner()
        clazz, score = model_runner.run_predict(
            parsed_args=MockParsedArgs(), encoded_dataset=encoded_dataset
        )

        assert clazz == "Readable"
        assert score == 0.9931080341339111

    def test_run_evaluate(self):
        # Mock the parsed arguments
        class MockParsedArgs:
            def __init__(self, save: str = self.output_dir):
                self.load = str(TOWARDS_MODEL)
                self.batch_size = 8
                self.save = save

        # Load the right data
        self.encoded_data = load_encoded_dataset(str(ENCODED_COMBINED_DIR))

        # Run the model runner
        model_runner = KerasModelRunner()
        model_runner.run_evaluate(
            parsed_args=MockParsedArgs(), encoded_data=self.encoded_data
        )
