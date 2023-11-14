import logging

from transformers import BertTokenizer

from readability_classifier.models.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)

DEFAULT_TOKEN_LENGTH = 512  # Maximum length of tokens for BERT
DEFAULT_ENCODE_BATCH_SIZE = 512


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
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Convert data to batches
        batches = [
            unencoded_dataset[i : i + DEFAULT_ENCODE_BATCH_SIZE]
            for i in range(0, len(unencoded_dataset), DEFAULT_ENCODE_BATCH_SIZE)
        ]

        # Log the number of batches to encode
        logging.info(f"Number of batches to encode: {len(batches)}")

        # Encode the batches
        encoded_batches = []
        for batch in batches:
            logging.info(f"Encoding batch: {len(encoded_batches) + 1}/{len(batches)}")
            encoded_batches.append(self._encode_batch(batch, tokenizer))

        # Flatten the encoded batches
        encoded_dataset = [sample for batch in encoded_batches for sample in batch]

        # Log the number of samples in the encoded dataset
        logging.info(f"Encoding done. Number of samples: {len(encoded_dataset)}")

        return ReadabilityDataset(encoded_dataset)

    def encode_text(self, text: str) -> dict:
        """
        Tokenizes and encodes the given text using the BERT tokenizer.
        :param text: The text to tokenize and encode.
        :return: A dictionary containing the encoded input_ids and attention_mask.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.token_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Log that the text was encoded
        logging.info("Text encoded.")

        return {
            "input_ids": encoding["input_ids"],
            "token_type_ids": encoding["token_type_ids"],  # Same as segment_ids
            "attention_mask": encoding["attention_mask"],
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
