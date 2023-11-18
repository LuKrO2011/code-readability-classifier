import logging
import re

from torch import Tensor
from transformers import BertTokenizer

from readability_classifier.models.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)

DEFAULT_TOKEN_LENGTH = 100  # Maximum length of tokens for BERT
DEFAULT_ENCODE_BATCH_SIZE = 500


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

    def encode_dataset(self, unencoded_dataset: list[dict]) -> ReadabilityDataset:
        """
        Encodes the given dataset with BERT.
        :param unencoded_dataset: The unencoded dataset.
        :return: The encoded dataset.
        """
        # Tokenize and encode the code snippets
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # Split identifiers in code snippets
        for sample in unencoded_dataset:
            sample["code_snippet"] = _split_identifiers(sample["code_snippet"])

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

        # Encode segment ids
        # for sample_encoded, sample_unencoded in zip(encoded_dataset,
        # unencoded_dataset):
        #     sample_encoded["token_type_ids"] = _calculate_segment_ids(
        #         sample_unencoded["code_snippet"])

        # Log the number of samples in the encoded dataset
        logging.info(f"Bert encoding done. Number of samples: {len(encoded_dataset)}")

        return ReadabilityDataset(encoded_dataset)

    def encode_text(self, text: str) -> dict:
        """
        Tokenizes and encodes the given text using the BERT tokenizer.
        :param text: The text to tokenize and encode.
        :return: A dictionary containing the encoded input_ids and attention_mask.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        text = _split_identifiers(text)
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.token_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Log successful encoding
        logging.info("Bert: Text encoded.")

        # Create own segment ids
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]  # _calculate_segment_ids(input_ids)
        attention_mask = encoding["attention_mask"]

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,  # Same as segment_ids
            "attention_mask": attention_mask,
        }

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
        token_type_ids = batch_encoding["token_type_ids"]  # Same as segment_ids
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


# TODO: This embedding causes an error when passed to Bert
def _calculate_segment_ids(text: str, length: int = 100, padding=0) -> Tensor:
    """
    Calculates the segment ids for the given code snippet.
    The resulting segment embedding is made up of sentence indexes representing which
    sentence every token is in.
    :param text: The code snippet.
    :param length: The exact length of the segment ids.
    :param padding: The padding to add, if the length of the segment ids is smaller than
    the given length.
    :return: The segment ids.
    """
    # Split the code snippet into sentences
    sentences = re.split(r"\.|\n", text)

    # Calculate the segment ids
    segment_ids = []
    for idx, sentence in enumerate(sentences):
        # Split the sentence into tokens
        tokens = sentence.split()

        # Add the segment ids for the tokens
        segment_ids += [idx] * len(tokens)

    # Pad the segment ids if too short
    segment_ids += [padding] * (length - len(segment_ids))

    # Remove segment ids if too long
    if len(segment_ids) > length:
        segment_ids = segment_ids[:length]

    # Convert the segment ids to an int tensor
    return Tensor(segment_ids).long()
