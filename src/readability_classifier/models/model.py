import logging

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

DEFAULT_TOKEN_LENGTH = 512  # Maximum length of tokens for BERT
DEFAULT_MODEL_BATCH_SIZE = 8  # Small - avoid CUDA out of memory errors on local machine
DEFAULT_ENCODE_BATCH_SIZE = 128


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
        # TODO: This is obsolete?
        return {
            "input_ids": self.data[idx]["input_ids"],
            "attention_mask": self.data[idx]["attention_mask"],
            "score": self.data[idx]["score"],
        }


class CNNModel(nn.Module):
    """
    A CNN model for code readability classification. The model consists of a Bert
    embedding layer, two convolutional layers, two max-pooling layers, two fully
    connected layers and a dropout layer.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialize the model. The number of classes is set to 1 for regression.
        Then 5 means very readable, 1 means very unreadable (Likert scale).
        :param num_classes: The number of classes.
        """
        super().__init__()
        self.num_classes = num_classes

        # Bert embedding
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 768))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1))

        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(8064, 128)  # 8 * 8064 = shape of x
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param input_ids:   Tensor of input_ids for the BERT model.
        :param attention_mask: Tensor of attention_mask for the BERT model.
        :return: The output of the model.
        """
        # Bert embedding
        x = self.bert(input_ids, attention_mask=attention_mask)

        # Convert the output of the Bert embedding to fitting shape for conv layers
        x = x[0].unsqueeze(1)

        # Apply convolutional and pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        # Flatten the output of the conv layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def load_raw_dataset(data_dir: str) -> list[dict]:
    """
    Loads the data from a dataset in the given directory as a list of dictionaries
    code_snippet, score.
    :param data_dir: The path to the directory containing the data.
    :return: A list of dictionaries.
    """
    dataset = load_from_disk(data_dir)
    return dataset.to_list()


def load_encoded_dataset(data_dir: str) -> list[dict[str, torch.Tensor]]:
    """
    Loads the BERT encoded data from a dataset in the given directory as a list of
    dictionaries containing torch.Tensors (input_ids, attention_mask, score).
    :param data_dir: The path to the directory containing the data.
    :return: A list of dictionaries containing the input_ids, attention_mask, and the
    score for the sample as torch.Tensors.
    """
    dataset = load_from_disk(data_dir)
    dataset_list = dataset.to_list()

    # TODO: Change data format of model to avoid long (and float32) conversion

    # Convert loaded data to torch.Tensors
    for sample in dataset_list:
        sample["input_ids"] = torch.tensor(sample["input_ids"]).long()
        sample["attention_mask"] = torch.tensor(sample["attention_mask"]).long()
        sample["score"] = torch.tensor(sample["score"], dtype=torch.float32)

    return dataset_list


def store_encoded_dataset(data: list[dict[str, torch.Tensor]], data_dir: str) -> None:
    """
    Stores the encoded data in the given directory.
    :param data: The encoded data.
    :param data_dir: The directory to store the encoded data in.
    :return: None
    """
    # Convert the encoded data to Hugging faces format
    HFDataset.from_list(data).save_to_disk(data_dir)


def encoded_data_to_dataloaders(
    encoded_data: list[dict[str, torch.Tensor]],
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

    def encode(self, unencoded_dataset: list[dict]) -> list[dict]:
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

        # Encode the batches
        encoded_batches = []
        for batch in batches:
            encoded_batches.append(self._encode_batch(batch, tokenizer))

        # Flatten the encoded batches
        return [sample for batch in encoded_batches for sample in batch]

    @staticmethod
    def tokenize_and_encode(
        text: str, token_length: int = DEFAULT_TOKEN_LENGTH
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes and encodes the given text using the BERT tokenizer.
        :param text: The text to tokenize and encode.
        :param token_length: The length of the encoded tokens.
        :return: A tuple containing the input_ids and the attention_mask.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # TODO: USE tokenizer(...) instead of .encode!!!!
        input_ids = tokenizer.encode_plus(
            text, add_special_tokens=True, truncation=True, max_length=token_length
        )

        # Create an attention_mask
        attention_mask = [1] * len(input_ids) + [0] * (token_length - len(input_ids))

        # Ensure the input_ids have a maximum length of MAX_TOKEN_LENGTH
        if len(input_ids) < token_length:
            # Pad the input_ids with zeros to match MAX_TOKEN_LENGTH
            input_ids += [0] * (token_length - len(input_ids))
        else:
            # If the input_ids exceed MAX_TOKEN_LENGTH, truncate them
            # TODO: Necessary? Already done by tokenizer?
            input_ids = input_ids[:token_length]

        # Convert to PyTorch tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        return input_ids, attention_mask

    # TODO: Methods static or not?
    def _encode_batch(self, batch: list[dict], tokenizer: BertTokenizer) -> list[dict]:
        """
        Tokenizes and encodes a batch of code snippets with BERT.
        :param batch: The batch of code snippets.
        :param tokenizer: The BERT tokenizer.
        :return: The encoded batch.
        """
        encoded_batch = []

        # Encode the code snippets using tokenizer.encode_plus(...)
        # TODO: Beside attention mask and padding, also use token_type_ids?
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
        attention_mask = batch_encoding["attention_mask"]

        # Create a dictionary for each sample in the batch
        for i in range(len(batch)):
            encoded_batch.append(
                {
                    "input_ids": input_ids[i],
                    "attention_mask": attention_mask[i],
                    "score": torch.tensor(batch[i]["score"], dtype=torch.float32),
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
        model_path: str = None,
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
        self.model = CNNModel(1)  # Set number of classes to 1 for regression
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _train_iteration(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor, attention_mask: torch.Tensor
    ) -> float:
        """
        Performs a single training iteration.
        :param x_batch: The input_ids of the batch.
        :param y_batch: The scores of the batch.
        :param attention_mask: The attention_mask of the batch.
        :return: The loss of the batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(x_batch, attention_mask)
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
                attention_mask = batch["attention_mask"].to(self.device)
                score = (
                    batch["score"].unsqueeze(1).to(self.device)
                )  # Add dimension for matching batch size

                loss = self._train_iteration(
                    input_ids, score, attention_mask=attention_mask
                )
                running_loss += loss

            logging.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"Loss: {running_loss / len(self.train_loader)}"
            )

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
                attention_mask = batch["attention_mask"].to(self.device)
                score = batch["score"].to(self.device)

                y_batch.append(score)
                predictions.append(self.model(input_ids, attention_mask))

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

    def load(self, path: str) -> None:
        """
        Loads the model from the given path.
        :param path: The path to load the model from.
        :return: None
        """
        self.model.load_state_dict(torch.load(path))

    def predict(self, code_snippet: str) -> float:
        """
        Predicts the readability of the given code snippet.
        :param code_snippet: The code snippet to predict the readability of.
        :return: The predicted readability.
        """
        self.model.eval()
        with torch.no_grad():
            input_ids, attention_mask = DatasetEncoder.tokenize_and_encode(code_snippet)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            prediction = self.model(input_ids, attention_mask)
            return prediction.item()
