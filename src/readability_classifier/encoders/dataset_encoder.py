import logging

from torch import Tensor

from readability_classifier.encoders.bert_encoder import BertEncoder
from readability_classifier.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)
from readability_classifier.encoders.image_encoder import VisualEncoder
from readability_classifier.encoders.matrix_encoder import MatrixEncoder


class DatasetEncoder(EncoderInterface):
    """
    A class for encoding code snippets as matrix, with bert and as image.
    Uses the MatrixEncoder, BertEncoder and VisualEncoder for encoding.
    The output is used by the model.
    """

    def __init__(self):
        """
        Initializes the DatasetEncoder.
        """
        self.matrix_encoder = MatrixEncoder()
        self.bert_encoder = BertEncoder()
        self.visual_encoder = VisualEncoder()

    def encode_text(self, code_text: str) -> ReadabilityDataset:
        """
        Encodes the given code snippet as a matrix, bert and image.
        :param code_text: The code snippet to encode.
        :return: The encoded code snippet.
        """
        matrix = self.matrix_encoder.encode_text(code_text)
        bert = self.bert_encoder.encode_text(code_text)
        image = self.visual_encoder.encode_text(code_text)

        # Log successful encoding
        logging.info("All: Encoding done.")

        return ReadabilityDataset(
            [{"matrix": matrix["matrix"], "bert": bert, "image": image["image"]}]
        )


def decode_score(score: Tensor) -> tuple[str, float]:
    """
    Decodes the given score to a tuple with class and score.
    :param score: The score to decode.
    :return: The decoded score.
    """
    score = score.item()
    return "Readable" if score > 0.5 else "Unreadable", score
