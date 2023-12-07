import logging
import re

import torch
from transformers import BertTokenizer

from src.readability_classifier.models.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
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

    def encode_dataset(
        self,
        unencoded_dataset: list[dict],
        own_segment_ids: bool = DEFAULT_OWN_SEGMENT_IDS,
    ) -> ReadabilityDataset:
        """
        Encodes the given dataset with BERT.
        If own_segment_ids is True, each line is considered a sentence.
        :param unencoded_dataset: The unencoded dataset.
        :param own_segment_ids: Whether to use own segment ids or not.
        :return: The encoded dataset.
        """
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # Add a special token "NEWLINE" to the vocabulary
        if own_segment_ids:
            tokenizer.add_tokens(NEWLINE_TOKEN)

        # Split identifiers in code snippets
        for sample in unencoded_dataset:
            sample["code_snippet"] = _split_identifiers(sample["code_snippet"])

            # Add a special token "NEWLINE" to the text
            if own_segment_ids:
                sample["code_snippet"] = _add_separators(
                    sample["code_snippet"], NEWLINE_TOKEN
                )

        # Convert data to batches
        batches = [
            unencoded_dataset[i : i + DEFAULT_ENCODE_BATCH_SIZE]
            for i in range(0, len(unencoded_dataset), DEFAULT_ENCODE_BATCH_SIZE)
        ]

        # Log the number of batches to encode
        logging.info(f"Bert: Number of batches to encode: {len(batches)}")

        # Encode the batches
        encoded_batches = []
        for batch in batches:
            logging.info(f"Encoding batch: {len(encoded_batches) + 1}/{len(batches)}")
            encoded_batches.append(self._encode_batch(batch, tokenizer))

        # Flatten the encoded batches
        encoded_dataset = [sample for batch in encoded_batches for sample in batch]

        # Calculate segment ids
        if own_segment_ids:
            newline_token_id = tokenizer.encode(NEWLINE_TOKEN)[1]
            for sample in encoded_dataset:
                segment_ids = _calculate_segment_ids(
                    sample["input_ids"].tolist(), newline_token_id
                )
                sample["segment_ids"] = torch.Tensor(segment_ids).long()

        # Calculate the position ids
        for sample in encoded_dataset:
            sample["position_ids"] = torch.arange(self.token_length).long()

        # Log the number of samples in the encoded dataset
        logging.info(f"Bert: Encoding done. Number of samples: {len(encoded_dataset)}")

        return ReadabilityDataset(encoded_dataset)

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

    def _encode_batch(self, batch: list[dict], tokenizer: BertTokenizer) -> list[dict]:
        """
        Tokenizes and encodes a batch of code snippets with BERT.
        :param batch: The batch of code snippets.
        :param tokenizer: The BERT tokenizer.
        :return: The encoded batch.
        """
        encoded_batch = []

        # Encode the code snippets batch
        batch_encoding = tokenizer.batch_encode_plus(
            [sample["code_snippet"] for sample in batch],
            add_special_tokens=True,
            truncation=True,
            max_length=self.token_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Extract input ids and attention mask from batch_encoding
        input_ids = batch_encoding["input_ids"]
        token_type_ids = batch_encoding["token_type_ids"]
        attention_mask = batch_encoding["attention_mask"]

        # Create a dictionary for each sample in the batch
        for i in range(len(batch)):
            encoded_batch.append(
                {
                    "input_ids": input_ids[i],
                    "token_type_ids": token_type_ids[i],
                    "attention_mask": attention_mask[i],
                }
            )

        return encoded_batch


def _split_identifiers(
    text: str, camel_case_regex=r"([a-z])([A-Z])", snake_case_regex=r"([a-z])(_)([a-z])"
):
    """
    Splits the identifiers in the given text.
    :param text: The text to split.
    :param camel_case_regex: The regex for camel case identifiers.
    :param snake_case_regex: The regex for snake case identifiers.
    :return: The text with split identifiers.
    """
    # Split camel case identifiers
    new_text = re.sub(camel_case_regex, r"\1 \2", text)

    # Split snake case identifiers
    new_text = re.sub(snake_case_regex, r"\1 \2 \3", new_text)

    return new_text


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
