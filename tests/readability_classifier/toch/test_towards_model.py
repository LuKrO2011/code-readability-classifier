import unittest

from src.readability_classifier.toch.towards_model import TowardsModel
from src.readability_classifier.utils.config import TowardsInput
from tests.readability_classifier.toch.extractors.test_semantic_extractor import (
    create_test_data as create_semantic_test_data,
)
from tests.readability_classifier.toch.extractors.test_structural_extractor import (
    create_test_data as create_structural_test_data,
)
from tests.readability_classifier.toch.extractors.test_visual_extractor import (
    create_test_data as create_visual_test_data,
)


def create_test_data():
    structural_input_data = create_structural_test_data()
    semantic_input = create_semantic_test_data()
    visual_input_data = create_visual_test_data()

    return TowardsInput(
        character_matrix=structural_input_data,
        bert=semantic_input,
        image=visual_input_data,
    )


class TestTowardsModel(unittest.TestCase):
    readability_model = TowardsModel.build_from_config()

    # TODO: Does not work
    def test_forward_pass(self):
        # Create test input data
        input_data = create_test_data()

        # Run the forward pass
        output = self.readability_model(input_data)

        # Check the output shape
        assert output.shape == (1, 1)
