import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from src.readability_classifier.utils.config import DEFAULT_MODEL_BATCH_SIZE


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
    dataset = load_from_disk(str(data_dir))
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


@dataclass
class Datasets:
    """
    A class for storing the different datasets:
    - Training Set (further split into train/validation during k-fold CV)
    - Test Set (completely separate for final model evaluation)
    Each dataset is a ReadabilityDataset.
    """

    train_set: ReadabilityDataset
    test_set: ReadabilityDataset


@dataclass
class Fold:
    """
    A class for storing the train and validation sets of a fold:
    - Training Set
    - Validation Set
    Each dataset is a ReadabilityDataset.
    """

    train_set: ReadabilityDataset
    val_set: ReadabilityDataset


def split_train_test(
    dataset: ReadabilityDataset,
) -> Datasets:
    """
    Splits the encoded data into Datasets. The datasets contain the training and test
    data.
    :param dataset: The encoded data.
    :return: The datasets containing the training and test data.
    """
    # Split data into training/validation and test data
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

    # Convert the split data to ReadabilityDataset
    train_dataset = ReadabilityDataset(train_data)
    test_dataset = ReadabilityDataset(test_data)

    # Log the number of samples in the training, validation, and test data
    logging.info(f"Training data: {len(train_dataset)} samples")
    logging.info(f"Test data: {len(test_dataset)} samples")

    return Datasets(test_set=test_dataset, train_set=train_dataset)


def split_train_val(
    train_dataset: ReadabilityDataset,
) -> Fold:
    """
    Splits the training data into a training and validation set. Used if no k-fold CV is
    used.
    :param train_dataset: The training data.
    :return: The training and validation sets.
    """
    # Split data into training and validation data
    train_data, val_data = train_test_split(
        train_dataset, test_size=0.1, random_state=42
    )

    # Convert the split data to ReadabilityDataset
    train_dataset = ReadabilityDataset(train_data)
    val_dataset = ReadabilityDataset(val_data)

    # Log the number of samples in the training, validation, and test data
    logging.info(f"Training data: {len(train_dataset)} samples")
    logging.info(f"Validation data: {len(val_dataset)} samples")

    return Fold(train_set=train_dataset, val_set=val_dataset)


def dataset_to_dataloader(
    dataset: ReadabilityDataset, batch_size: int = DEFAULT_MODEL_BATCH_SIZE
) -> DataLoader:
    """
    Converts a readability dataset to a data loader.
    :param dataset: The dataset.
    :param batch_size: The batch size.
    :return: The data loader.
    """
    # Create data loaders for training, validation, and test sets
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Log the number of samples in the training, validation, and test data
    logging.info(f"Training data: {len(dataset)} samples")

    return loader


def split_k_fold(dataset: ReadabilityDataset, k_fold: int = 0) -> list[Fold]:
    """
    Splits the training data into k folds.
    :param dataset: The training data.
    :param k_fold: The number of folds.
    :return: The folds.
    """
    if k_fold == 0:
        return [split_train_val(dataset)]

    # Split data into k folds
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    # Convert the split data to ReadabilityDataset
    folds = []
    for train_index, val_index in kf.split(dataset):
        train_dataset = ReadabilityDataset([dataset[i] for i in train_index])
        val_dataset = ReadabilityDataset([dataset[i] for i in val_index])
        folds.append(Fold(train_set=train_dataset, val_set=val_dataset))

    # Log the number of folds and the number of samples in the first fold
    logging.info(f"Number of folds: {k_fold}")
    logging.info(f"Training data in first fold: {len(folds[0].train_set)} samples")
    logging.info(f"Validation data in first fold: {len(folds[0].val_set)} samples")

    return folds
