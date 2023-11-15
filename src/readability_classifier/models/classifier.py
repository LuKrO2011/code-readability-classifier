import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from readability_classifier.models.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.utils.config import DEFAULT_MODEL_BATCH_SIZE
from readability_classifier.utils.utils import save_content_to_file
from src.readability_classifier.models.towards_model import TowardsModel


class CodeReadabilityClassifier:
    """
    A code readability classifier based on a CNN model. The model can be used to predict
    the readability of a code snippet.
    The model is trained on code snippets and their corresponding scores.
    The model uses the following features:
    - Structural features (ASCII matrix)
    - Semantic features (Bert embedding)
    - Visual features (Image of the code, where words are replaced by color bars
    depending on their token type)
    """

    def __init__(
        self,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        validation_loader: DataLoader = None,
        store_dir: Path = None,
        batch_size: int = DEFAULT_MODEL_BATCH_SIZE,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """
        Initializes the classifier.
        :param train_loader: The data loader for the training data.
        :param test_loader: The data loader for the test data.
        :param validation_loader: The data loader for the validation data.
        :param batch_size: The batch size.
        :param num_epochs: The number of epochs.
        :param learning_rate: The learning rate.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.store_dir = store_dir
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
        self.model = TowardsModel.build_from_config()
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _fit_batch(
        self,
        matrix: Tensor,
        input_ids: Tensor,
        token_type_ids: Tensor,
        image: Tensor,
        y_batch: Tensor,
    ) -> float:
        """
        Performs a single training iteration.
        :param matrix: The matrix of the batch.
        :param input_ids: The input ids of the batch.
        :param token_type_ids: The token type ids of the batch.
        :param image: The image of the batch.
        :param y_batch: The scores of the batch.
        :return: The loss of the batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(matrix, input_ids, token_type_ids, image)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self):
        """
        Trains the model.
        :return: None
        """
        if self.train_loader is None:
            raise ValueError("No training data provided.")

        if self.test_loader is None:
            raise ValueError("No test data provided.")

        train_stats = TrainStats(0, [])
        best_test_loss = float("inf")

        for epoch in range(self.num_epochs):
            train_loss = self._fit_epoch()
            test_loss = self._eval_epoch()

            # Update stats
            epoch_stats = EpochStats(epoch + 1, train_loss, test_loss)
            train_stats.epoch_stats.append(epoch_stats)

            # Log the loss
            logging.info(
                f"Epoch {epoch + 1:02}/{self.num_epochs:02}\n"
                f"Train loss: {train_loss:.4f}\n"
                f"Train PPL:  {math.exp(train_loss):7.4f}\n"
                f"Test  loss: {test_loss:.4f}\n"
                f"Test  PPL:  {math.exp(test_loss):7.4f}"
            )

            # Save the model
            self.store(epoch=epoch + 1)

            # Update best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                train_stats.best_epoch = epoch + 1
                self.store(path=self.store_dir / Path("best_model.pt"))

        # Save the training stats
        save_content_to_file(
            train_stats.to_json(),
            self.store_dir / Path("train_stats.json"),
        )

        logging.info("Training done.")

    def _fit_epoch(self) -> float:
        """
        Trains a single epoch.
        :return: The train loss of the epoch.
        """
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            matrix = batch["matrix"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            image = batch["image"].to(self.device)
            score = (
                batch["score"].unsqueeze(1).to(self.device)
            )  # Add dimension for matching batch size

            loss = self._fit_batch(
                matrix=matrix,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                image=image,
                y_batch=score,
            )
            train_loss += loss
        return train_loss / len(self.train_loader)

    def _eval_epoch(self) -> float:
        """
        Evaluates the model on the test data.
        :return: The validation loss.
        """
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            # Iterate through the test loader to evaluate the model
            for batch in self.test_loader:
                matrix = batch["matrix"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                image = batch["image"].to(self.device)
                score = batch["score"].to(self.device)

                loss = self._eval_batch(
                    matrix=matrix,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    image=image,
                    y_batch=score,
                )

                valid_loss += loss

        return valid_loss / len(self.test_loader)

    def _eval_batch(
        self,
        matrix: Tensor,
        input_ids: Tensor,
        token_type_ids: Tensor,
        image: Tensor,
        y_batch: Tensor,
    ) -> float:
        """
        Evaluates a single batch of the test loader.
        :param input_ids: The input ids of the batch.
        :param token_type_ids: The token type ids of the batch.
        :param image: The image of the batch.
        :param y_batch: The scores of the batch.
        :return: The loss of the batch.
        """
        outputs = self.model(matrix, input_ids, token_type_ids, image)
        loss = self.criterion(outputs, y_batch)
        return loss.item()

    def evaluate(self) -> None:
        """
        Evaluates the model on the validation data.
        :return: The MSE of the model on the validation data.
        """
        if self.validation_loader is None:
            raise ValueError("No test data provided.")

        self.model.eval()
        with torch.no_grad():
            y_batch = []  # True scores
            predictions = []  # List to store model predictions

            # Iterate through the test loader to evaluate the model
            for batch in self.validation_loader:
                matrix = batch["matrix"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                image = batch["image"].to(self.device)
                score = batch["score"].to(self.device)

                y_batch.append(score)
                predictions.append(self.model(matrix, input_ids, token_type_ids, image))

        # Concatenate the lists of tensors to create a single tensor
        y_batch = torch.cat(y_batch, dim=0)
        predictions = torch.cat(predictions, dim=0)

        # Compute Mean Squared Error (MSE) using PyTorch
        mse = torch.mean((y_batch - predictions) ** 2).item()

        # Log the MSE
        logging.info(f"MSE: {mse}")

        return mse

    def store(self, path: str = None, epoch: int = None) -> None:
        """
        Stores the model at the given path.
        :param path: The path to store the model.
        :param epoch: The epoch to store the model at.
        :return: None
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if path is None and epoch is None:
            path = self.store_dir / Path(f"model_{current_time}.pt")
        elif path is None:
            path = self.store_dir / Path(f"model_{current_time}_{epoch}.pt")

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
        matrix = encoded_text["matrix"]
        input_ids = encoded_text["input_ids"]
        token_type_ids = encoded_text["token_type_ids"]
        image = encoded_text["image"]

        # Predict the readability
        with torch.no_grad():
            matrix = matrix.to(self.device)
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            image = image.to(self.device)
            prediction = self.model(matrix, input_ids, token_type_ids, image)
            return prediction.item()


@dataclass(frozen=True, eq=True)
class EpochStats:
    """
    Data class for epoch stats.
    """

    epoch: int
    train_loss: float
    test_loss: float

    def to_json(self) -> str:
        """
        Convert to json.
        :return: Returns the json string.
        """
        return json.dumps(asdict(self))


@dataclass(eq=True)
class TrainStats:
    """
    Data class for training stats.
    """

    best_epoch: int
    epoch_stats: list[EpochStats]
    start_time: int = int(time())
    end_time: int = int(time())

    def to_json(self) -> str:
        """
        Convert to json.
        :return: Returns the json string.
        """
        # Update end time every time when stats are saved
        self.end_time = int(time())
        return json.dumps(asdict(self))
