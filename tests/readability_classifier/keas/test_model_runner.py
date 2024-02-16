from readability_classifier.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.encoders.dataset_utils import load_encoded_dataset
from readability_classifier.utils.utils import read_content_of_file
from src.readability_classifier.keas.model_runner import KerasModelRunner
from tests.readability_classifier.utils.utils import (
    ENCODED_BW_DIR,
    TOWARDS_CODE_SNIPPET,
    TOWARDS_MODEL,
    DirTest,
)


class TestKerasModelRunner(DirTest):
    def setUp(self):
        super().setUp()
        self.encoded_data = load_encoded_dataset(ENCODED_BW_DIR)

    def test_run_predict(self):
        # Mock the parsed arguments
        class MockParsedArgs:
            def __init__(self):
                self.model = TOWARDS_MODEL

        # Get the encoded data
        code = read_content_of_file(TOWARDS_CODE_SNIPPET)
        encoded_data = DatasetEncoder().encode_text(code)

        # Run the model runner
        model_runner = KerasModelRunner()
        clazz, score = model_runner.run_predict(
            parsed_args=MockParsedArgs(), encoded_data=encoded_data
        )

        assert clazz == "Unreadable"
        assert score == 0.3686406910419464
