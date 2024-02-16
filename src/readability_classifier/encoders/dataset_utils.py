import logging

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

DEFAULT_MODEL_BATCH_SIZE = 8


class ReadabilityDataset(Dataset):
    """
    A class for storing the data in a dataset. The content of the dataset is a list of
    dictionaries containing encoded code snippet variants and/or their scores.
    """

    def __init__(self, data: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]]):
        """
        Initialize the dataset with a dictionary containing data samples.
        :param data: A list of dictionaries containing the data samples.
        """
        self.data = data

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Return a sample from the dataset by its index. The sample is a dictionary
        containing encoded code snippet variants and/or their scores.
        :param idx: The index of the sample.
        :return: The dictionary containing the sample.
        """
        return self.data[idx]

    def to_list(self) -> list[dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        """
        Return the dataset as a list.
        :return: A list containing the data samples.
        """
        return self.data

    def split(self, parts: int) -> list["ReadabilityDataset"]:
        """
        Splits the dataset into #parts datasets.
        :param parts: The number of parts to split the dataset into.
        :return: A list of datasets.
        """
        # Get the data as list
        data = self.to_list()

        # Split the data into #parts parts
        split_data = []
        for i in range(parts):
            split_data.append(ReadabilityDataset(data[i::parts]))

        return split_data


class EncoderInterface:
    """
    An interface for encoding the code of the dataset.
    """

    def encode_dataset(self, unencoded_dataset: list[dict]) -> ReadabilityDataset:
        """
        Encodes the given dataset.
        :param unencoded_dataset: The unencoded dataset.
        :return: The encoded dataset.
        """
        raise NotImplementedError

    def encode_text(self, text: str) -> dict:
        """
        Encodes the given text.
        :param text: The text to encode.
        :return: The encoded text.
        """
        raise NotImplementedError


def load_raw_dataset(data_dir: str) -> list[dict]:
    """
    Loads the data from a dataset in the given directory as a list of dictionaries
    code_snippet, score.
    :param data_dir: The path to the directory containing the data.
    :return: A list of dictionaries.
    """
    dataset = load_from_disk(data_dir)
    if "train" in dataset:
        dataset = dataset["train"]
    return dataset.to_list()


# TODO: Move conversion to tensor to classifier (specific for pytorch)
def load_encoded_dataset(data_dir: str) -> ReadabilityDataset:
    """
    Loads the encoded data (with DatasetEncoder) from a dataset in the given directory
    as a ReadabilityDataset.
    :param data_dir: The path to the directory containing the data.
    :return: A ReadabilityDataset.
    """
    dataset = load_from_disk(data_dir)
    dataset_list = dataset.to_list()

    # Convert loaded data to torch.Tensors
    for sample in dataset_list:
        sample["matrix"] = torch.tensor(
            sample["matrix"], dtype=torch.float32
        )  # Why not int?
        # TODO: The following 4 should be own dic with 4th optional
        sample["bert"]["input_ids"] = torch.tensor(
            sample["bert"]["input_ids"], dtype=torch.long  # Why not int? Why long?
        )  # Why not int?
        if "token_type_ids" in sample["bert"]:
            sample["bert"]["token_type_ids"] = torch.tensor(
                sample["bert"]["token_type_ids"],
                dtype=torch.long
                # Why not int? Why long?
            )
        if "attention_mask" in sample["bert"]:
            sample["bert"]["attention_mask"] = torch.tensor(
                sample["bert"]["attention_mask"],
                dtype=torch.long
                # Why not int? Why long?
            )
        if "segment_ids" in sample["bert"]:
            sample["bert"]["segment_ids"] = torch.tensor(
                sample["bert"]["segment_ids"],
                dtype=torch.long
                # Why not int? Why long?
            )
        if "position_ids" in sample["bert"]:
            sample["bert"]["position_ids"] = torch.tensor(
                sample["bert"]["position_ids"],
                dtype=torch.long
                # Why not int? Why long?
            )
        sample["image"] = torch.tensor(sample["image"], dtype=torch.float32)
        sample["score"] = torch.tensor(sample["score"], dtype=torch.float32)

    # Log the number of samples in the dataset
    logging.info(f"Loaded {len(dataset_list)} samples from {data_dir}")

    return ReadabilityDataset(dataset_list)
