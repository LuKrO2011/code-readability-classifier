from src.readability_classifier.keas.model_runner import KerasModelRunner
from src.readability_classifier.main import _run_predict
from tests.readability_classifier.utils.utils import (
    BW_SNIPPET_1,
    TOWARDS_MODEL,
    DirTest,
)


class TestRunMain(DirTest):

    def test_run_predict(self):
        class MockParsedArgs:
            def __init__(self):
                self.input = BW_SNIPPET_1
                self.model = TOWARDS_MODEL

        parsed_args = MockParsedArgs()
        model_runner = KerasModelRunner()

        clazz, score = _run_predict(parsed_args, model_runner)

        assert clazz == "Readable"
        assert score == 0.7045395374298096
