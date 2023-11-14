import logging

import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from readability_classifier.utils.config import DEFAULT_MODEL_BATCH_SIZE


class ReadabilityDataset(Dataset):
    """
    A class for storing the data in a dataset. The content of the dataset is a list of
    dictionaries containing encoded code snippet variants and/or their scores.
    """

    def __init__(self, data: list[dict[str, torch.Tensor]]):
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a sample from the dataset by its index. The sample is a dictionary
        containing encoded code snippet variants and/or their scores.
        :param idx: The index of the sample.
        :return: The dictionary containing the sample.
        """
        return self.data[idx]

    def to_list(self) -> list[dict]:
        """
        Return the dataset as a list.
        :return: A list containing the data samples.
        """
        return self.data


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
    return dataset.to_list()


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
        sample["matrix"] = torch.tensor(sample["matrix"], dtype=torch.float32)
        sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)
        sample["token_type_ids"] = torch.tensor(
            sample["token_type_ids"], dtype=torch.long
        )
        sample["image"] = torch.tensor(sample["image"], dtype=torch.float32)
        sample["score"] = torch.tensor(sample["score"], dtype=torch.float32)

    # Log the number of samples in the dataset
    logging.info(f"Loaded {len(dataset_list)} samples from {data_dir}")

    return ReadabilityDataset(dataset_list)


def store_encoded_dataset(data: ReadabilityDataset, data_dir: str) -> None:
    """
    Stores the encoded data in the given directory.
    :param data: The encoded data.
    :param data_dir: The directory to store the encoded data in.
    :return: None
    """
    # Convert the encoded data to Hugging faces format
    HFDataset.from_list(data.to_list()).save_to_disk(data_dir)

    # Log the number of samples stored
    logging.info(f"Stored {len(data)} samples in {data_dir}")


def encoded_data_to_dataloaders(
    encoded_data: ReadabilityDataset,
    batch_size: int = DEFAULT_MODEL_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Converts the encoded data to a training and test data loader.
    :param encoded_data: The encoded data.
    :param batch_size: The batch size.
    :return: A tuple containing the training and test data loader.
    """
    # Split data into training and test data
    train_data, test_data = train_test_split(
        encoded_data, test_size=0.2, random_state=42
    )

    # Convert the split data to a ReadabilityDataset
    train_dataset = ReadabilityDataset(train_data)
    test_dataset = ReadabilityDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Log the number of samples in the training and test data
    logging.info(f"Training data: {len(train_dataset)} samples")
    logging.info(f"Test data: {len(test_dataset)} samples")

    return train_loader, test_loader
