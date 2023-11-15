from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from readability_classifier.models.base_classifier import BaseClassifier
from readability_classifier.models.structural_model import StructuralModel
from readability_classifier.utils.config import (
    DEFAULT_MODEL_BATCH_SIZE,
    StructuralInput,
)


class StructuralClassifier(BaseClassifier):
    """
    A code readability classifier based on a CNN model. The model can be used to predict
    the readability of a code snippet.
    The model is trained on code snippets and their corresponding scores. The model uses
    the structural features (ASCII matrix) of the code snippets.
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
        self.model = StructuralModel.build_from_config()
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        super().__init__(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=validation_loader,
            store_dir=store_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )

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
                inp=StructuralInput(matrix),
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
                    inp=StructuralInput(matrix),
                    y_batch=score,
                )

                valid_loss += loss

        return valid_loss / len(self.test_loader)

    def evaluate(self) -> None:
        """
        Evaluates the model on the validation data.
        :return: The MSE of the model on the validation data.
        """
        raise NotImplementedError("Structural classifier does not support evaluation.")
