import os
import unittest

from src.readability_classifier.keras.keras_model_runner import KerasModelRunner
from src.readability_classifier.main import _run_encode, _run_train
from tests.readability_classifier.utils.utils import ENCODED_BW_DIR, RAW_BW_DIR, DirTest


class TestRunMain(DirTest):
    def test_run_encode(self):
        class MockParsedArgs:
            def __init__(self, save: str = self.output_dir):
                self.input = RAW_BW_DIR
                self.save = save
                self.intermediate = save

        parsed_args = MockParsedArgs()

        _run_encode(parsed_args)

        assert len(os.listdir(self.output_dir)) != 0

    @unittest.skip("Takes to long.")
    def test_run_train(self):
        class MockParsedArgs:
            def __init__(self, save: str = self.output_dir):
                self.model = None
                self.input = ENCODED_BW_DIR
                self.encoded = True
                self.save = save
                self.intermediate = None
                self.evaluate = True
                self.k_fold = 2
                self.batch_size = 2
                self.epochs = 2
                self.learning_rate = 0.0015

        parsed_args = MockParsedArgs()
        model_runner = KerasModelRunner()

        _run_train(parsed_args, model_runner)

        assert len(os.listdir(self.output_dir)) != 0

    @unittest.skip("Not implemented yet.")
    def test_run_evaluate(self):
        pass

    @unittest.skip("Not implemented yet.")
    def test_run_predict(self):
        pass
