import logging

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from src.readability_classifier.models.semantic_extractor import SemanticExtractor

DEFAULT_TOKEN_LENGTH = 512  # Maximum length of tokens for BERT
DEFAULT_MODEL_BATCH_SIZE = 8  # Small - avoid CUDA out of memory errors on local machine
DEFAULT_ENCODE_BATCH_SIZE = 512


class ReadabilityDataset(Dataset):
    def __init__(self, data: list[dict[str, torch.Tensor]]):
        """
        Initialize the dataset with a dictionary containing data samples.
        Each data sample is a dict containing the input_ids, attention_mask and the
        score for the sample.
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
        containing the input_ids, attention_mask and the score for the sample.
        :param idx: The index of the sample.
        :return: A dictionary containing the input_ids, attention_mask and the score
        for the sample.
        """
        return self.data[idx]

    def to_list(self) -> list[dict]:
        """
        Return the dataset as a list.
        :return: A list containing the data samples.
        """
        return self.data


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
    Loads the BERT encoded data from a dataset in the given directory as a
    ReadabilityDataset.
    :param data_dir: The path to the directory containing the data.
    :return: A ReadabilityDataset.
    """
    dataset = load_from_disk(data_dir)
    dataset_list = dataset.to_list()

    # Convert loaded data to torch.Tensors
    for sample in dataset_list:
        sample["input_ids"] = torch.tensor(sample["input_ids"])
        sample["token_type_ids"] = torch.tensor(sample["token_type_ids"])
        sample["attention_mask"] = torch.tensor(sample["attention_mask"])
        sample["score"] = torch.tensor(sample["score"])

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


class DatasetEncoder:
    """
    A class for encoding the code of the dataset with BERT.
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
            "token_type_ids": encoding["token_type_ids"],
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
        token_type_ids = batch_encoding["token_type_ids"]
        attention_mask = batch_encoding["attention_mask"]

        # Create a dictionary for each sample in the batch
        for i in range(len(batch)):
            encoded_batch.append(
                {
                    "input_ids": input_ids[i],
                    "token_type_ids": token_type_ids[i],
                    "attention_mask": attention_mask[i],
                    "score": torch.tensor(batch[i]["score"]),
                }
            )

        return encoded_batch


class CodeReadabilityClassifier:
    """
    A code readability classifier based on a CNN model. The model is trained on code
    snippets and their corresponding scores. The code snippets are tokenized and
    encoded using the BERT tokenizer. The model is trained on the encoded code
    snippets and their scores.
    """

    def __init__(
        self,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        batch_size: int = DEFAULT_MODEL_BATCH_SIZE,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """
        Initializes the classifier.
        :param train_loader: The data loader for the training data.
        :param test_loader: The data loader for the test data.
        :param batch_size: The batch size.
        :param num_epochs: The number of epochs.
        :param learning_rate: The learning rate.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up the model on initialization
        self._setup_model()

    def _setup_model(self):
        """
        Sets up the model. This includes initializing the model, the loss function and
        the optimizer.
        :return: None
        """
        self.model = SemanticExtractor(1)  # Set number of classes to 1 for regression
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _train_iteration(
        self,
        x_batch: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> float:
        """
        Performs a single training iteration.
        :param x_batch: The input_ids of the batch.
        :param token_type_ids: The token_type_ids of the batch.
        :param attention_mask: The attention_mask of the batch.
         :param y_batch: The scores of the batch.
        :return: The loss of the batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(x_batch, token_type_ids, attention_mask)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        """
        Trains the model.
        :return: None
        """
        if self.train_loader is None:
            raise ValueError("No training data provided.")

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            # Iterate over the training dataset in mini-batches
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                score = (
                    batch["score"].unsqueeze(1).to(self.device)
                )  # Add dimension for matching batch size

                loss = self._train_iteration(
                    x_batch=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    y_batch=score,
                )
                running_loss += loss

            logging.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"Loss: {running_loss / len(self.train_loader)}"
            )

        logging.info("Training done.")

    def evaluate(self) -> None:
        """
        Evaluates the model.
        :return: None
        """
        if self.test_loader is None:
            raise ValueError("No test data provided.")

        self.model.eval()
        with torch.no_grad():
            y_batch = []  # True scores
            predictions = []  # List to store model predictions

            # Iterate through the test loader to evaluate the model
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                score = batch["score"].to(self.device)

                y_batch.append(score)
                predictions.append(
                    self.model(input_ids, token_type_ids, attention_mask)
                )

            # Concatenate the lists of tensors to create a single tensor
            y_batch = torch.cat(y_batch, dim=0)
            predictions = torch.cat(predictions, dim=0)

            # Compute Mean Squared Error (MSE) using PyTorch
            mse = torch.mean((y_batch - predictions) ** 2).item()

            # Log the MSE
            logging.info(f"MSE: {mse}")

    def store(self, path: str) -> None:
        """
        Stores the model at the given path.
        :param path: The path to store the model.
        :return: None
        """
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model stored at {path}")

    def load(self, path: str) -> None:
        """
        Loads the model from the given path.
        :param path: The path to load the model from.
        :return: None
        """
        self.model.load_state_dict(torch.load(path))
        logging.info(f"Model loaded from {path}")

    def predict(self, code_snippet: str) -> float:
        """
        Predicts the readability of the given code snippet.
        :param code_snippet: The code snippet to predict the readability of.
        :return: The predicted readability.
        """
        self.model.eval()

        # Encode the code snippet
        encoder = DatasetEncoder()
        encoded_text = encoder.encode_text(code_snippet)
        input_ids = encoded_text["input_ids"]
        token_type_ids = encoded_text["token_type_ids"]
        attention_mask = encoded_text["attention_mask"]

        # Predict the readability
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            prediction = self.model(input_ids, token_type_ids, attention_mask)
            return prediction.item()
