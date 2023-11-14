import logging

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.readability_classifier.models.readability_model import ReadabilityModel

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
        self.model = ReadabilityModel()
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _train_iteration(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> float:
        """
        Performs a single training iteration.
        :param x_batch: The input of the batch.
         :param y_batch: The scores of the batch.
        :return: The loss of the batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(x_batch)
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
                # TODO: Update to use the correct keys
                input_ids = batch["input_ids"].to(self.device)
                score = (
                    batch["score"].unsqueeze(1).to(self.device)
                )  # Add dimension for matching batch size

                loss = self._train_iteration(x_batch=input_ids, y_batch=score)
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
                # TODO: Update to use the correct keys
                input_ids = batch["input_ids"].to(self.device)
                score = batch["score"].to(self.device)
                y_batch.append(score)
                predictions.append(self.model(input_ids))

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
        # TODO: Generate image etc.
        input_ids = None
        token_type_ids = None
        attention_mask = None

        # Predict the readability
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            prediction = self.model(input_ids, token_type_ids, attention_mask)
            return prediction.item()
