import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from readability_classifier.models.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.utils.config import DEFAULT_MODEL_BATCH_SIZE
from src.readability_classifier.models.model import ReadabilityModel


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
                matrix = batch["matrix"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                image = batch["image"].to(self.device)
                score = (
                    batch["score"].unsqueeze(1).to(self.device)
                )  # Add dimension for matching batch size

                loss = self._train_iteration(
                    matrix=matrix,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    image=image,
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
