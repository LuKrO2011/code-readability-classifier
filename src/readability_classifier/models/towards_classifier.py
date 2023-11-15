import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from readability_classifier.models.base_classifier import BaseClassifier
from readability_classifier.utils.config import DEFAULT_MODEL_BATCH_SIZE, TowardsInput
from src.readability_classifier.models.towards_model import TowardsModel


class TowardsClassifier(BaseClassifier):
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
        num_epochs: int = 20,
        learning_rate: float = 0.0015,
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
        model = TowardsModel.build_from_config()
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=validation_loader,
            store_dir=store_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )

    def _fit_batch(self, inp: TowardsInput, y_batch: Tensor) -> float:
        """
        Performs a single training iteration.
        :param inp: The input of the model as batch.
        :return: The loss of the batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(inp)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _fit_epoch(self) -> float:
        """
        Trains a single epoch.
        :return: The train loss of the epoch.
        """
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            matrix, input_ids, token_type_ids, image, score = self._extract(batch)

            loss = self._fit_batch(
                inp=TowardsInput(matrix, input_ids, token_type_ids, image),
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
                matrix, input_ids, token_type_ids, image, score = self._extract(batch)

                loss = self._eval_batch(
                    inp=TowardsInput(matrix, input_ids, token_type_ids, image),
                    y_batch=score,
                )

                valid_loss += loss

        return valid_loss / len(self.test_loader)

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
                matrix, input_ids, token_type_ids, image, score = self._extract(batch)

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
