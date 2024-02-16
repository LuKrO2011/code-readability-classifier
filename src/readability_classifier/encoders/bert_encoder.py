import logging
import re

import torch
from transformers import BertTokenizer

from readability_classifier.encoders.dataset_utils import (
    EncoderInterface,
)

DEFAULT_TOKEN_LENGTH = 100  # Maximum length of tokens for BERT
DEFAULT_ENCODE_BATCH_SIZE = 500  # Number of samples to encode at once
NEWLINE_TOKEN = "[NL]"  # Special token for new lines
DEFAULT_OWN_SEGMENT_IDS = False  # Whether to use own segment ids or not


class BertEncoder(EncoderInterface):
    """
    A class for encoding code snippets with BERT.
    The output is used by the SemanticExtractor.
    """

    def __init__(self, token_length: int = DEFAULT_TOKEN_LENGTH):
        """
        Initializes the DatasetEncoder.
        """
        self.token_length = token_length

    def encode_text(
        self, text: str, own_segment_ids: bool = DEFAULT_OWN_SEGMENT_IDS
    ) -> dict:
        """
        Tokenizes and encodes the given text using the BERT tokenizer.
        If own_segment_ids is True, each line is considered a sentence.
        :param text: The text to tokenize and encode.
        :param own_segment_ids: Whether to use own segment ids or not.
        :return: A dictionary containing the encoded input_ids and attention_mask.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # Add a special token "NEWLINE" to the vocabulary
        if own_segment_ids:
            tokenizer.add_tokens(NEWLINE_TOKEN)
            text = _add_separators(text, NEWLINE_TOKEN)

        # Tokenize the text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.token_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Calculate segment ids
        if own_segment_ids:
            newline_token_id = tokenizer.encode(NEWLINE_TOKEN)[1]
            segment_ids = _calculate_segment_ids(
                encoding["input_ids"].tolist()[0], newline_token_id
            )
            encoding["segment_ids"] = torch.Tensor(segment_ids).long().unsqueeze(0)

        encoding["position_ids"] = torch.arange(self.token_length).long().unsqueeze(0)

        # Log successful encoding
        logging.info("Bert: Text encoded.")

        return encoding


def _calculate_segment_ids(input_ids: list[int], sep_token_id: int) -> list[int]:
    """
    Calculates the segment ids for the given code snippet.
    The resulting segment embedding is made up of sentence indexes representing which
    sentence every token is in. Each line is considered a sentence.
    :param input_ids: The encoded lines of the code snippet.
    :param: sep_token_id: The id of the separator token.
    :return: The segment ids.
    """
    segment_ids = []

    # Calculate the segment ids
    line = 0
    for token_id in input_ids:
        # Add the segment ids for the tokens
        segment_ids.append(line)

        # If the token is a separator token, increase the line number
        if token_id == sep_token_id:
            line += 1

    return segment_ids


def _add_separators(text: str, sep: str = "[SEP]") -> str:
    """
    Adds separators to the given text for each new line.
    :param text: The text to add separators to.
    :param sep: The separator to add.
    :return: The text with separators.
    """
    # Split the text into sentences
    sentences = re.split(r"\n", text)

    # Remove empty sentences (e.g. empty lines, only spaces or tabs)
    sentences = [sentence for sentence in sentences if sentence.strip() != ""]

    # Add separators to the sentences
    sentences = [sentence + " " + sep for sentence in sentences]

    # Join the sentences to a text again
    return "\n".join(sentences)
