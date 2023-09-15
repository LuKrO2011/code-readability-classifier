import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

TOKEN_LENGTH = 512  # Maximum length of tokens for BERT
DEFAULT_BATCH_SIZE = 8  # Small to avoid CUDA out of memory errors on local machine


class ReadabilityDataset(Dataset):
    def __init__(self, data_dict: dict[str, tuple[torch.Tensor, torch.Tensor, float]]):
        """
        Initialize the dataset with a dictionary containing data samples.

        Args:
            data_dict (dict): A dictionary where keys are sample names and values are
            tuples (input_ids, attention_mask, aggregated_scores).
        """
        self.data_dict = data_dict
        self.names = list(data_dict.keys())

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        """
        Return a sample from the dataset by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (dict): A dictionary containing the following keys:
                - 'input_ids': Tensor of input_ids for the BERT model.
                - 'attention_mask': Tensor of attention_mask for the BERT model.
                - 'scores': Tensor of aggregated_scores for the sample.
        """
        # TODO: Check where to convert data into tensors. Maybe here instead?
        name = self.names[idx]
        input_ids, attention_mask, scores = self.data_dict[name]

        # Remove the dimension of size 1
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scores": torch.tensor(scores, dtype=torch.float),
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


class CsvFolderDataLoader:
    """
    A data loader for loading data from a CSV file and the corresponding code snippets.
    TODO: Add hierarchy for different dataset formats?
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initializes the data loader.
        :param batch_size: The batch size.
        """
        self.batch_size = batch_size

    def load(self, csv: str, data_dir: str) -> tuple[DataLoader, DataLoader]:
        """
        Loads the data and prepares it for training and evaluation.
        :param csv: Path to the CSV file containing the scores.
        :param data_dir: Path to the directory containing the code snippets.
        :return: A tuple containing the training and test data loaders.
        """
        # Load data
        aggregated_scores, code_snippets = self._load_from_storage(csv, data_dir)

        # Tokenize and encode code snippets
        embeddings = {}
        for name, snippet in code_snippets.items():
            input_ids, attention_mask = self._tokenize_and_encode(snippet)
            embeddings[name] = (input_ids, attention_mask)

        # Combine embeddings (x) and scores (y) into a dictionary
        data = {}
        for name, (input_ids, attention_mask) in embeddings.items():
            data[name] = (input_ids, attention_mask, aggregated_scores[name])

        # Split into training and test data
        names = list(data.keys())
        train_names, test_names = train_test_split(
            names, test_size=0.2, random_state=42
        )
        train_data = {name: data[name] for name in train_names}
        test_data = {name: data[name] for name in test_names}

        # Convert the split data to a ReadabilityDatasets
        train_dataset = ReadabilityDataset(train_data)
        test_dataset = ReadabilityDataset(test_data)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, test_loader

    def _load_from_storage(self, csv: str, data_dir: str) -> tuple[dict, dict]:
        """
        Loads the data from the CSV file and the code snippets from the files.
        :param csv: The path to the CSV file containing the scores.
        :param data_dir: The path to the directory containing the code snippets.
        :return: A tuple containing the mean scores and the code snippets.
        """
        mean_scores = self._load_mean_scores(csv)
        code_snippets = self._load_code_snippets(data_dir)

        return mean_scores, code_snippets

    def _load_code_snippets(self, data_dir: str) -> dict:
        """
        Loads the code snippets from the files to a dictionary. The file names are used
        as keys and the code snippets as values.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The code snippets as a dictionary.
        """
        code_snippets = {}

        # Iterate through the files in the directory
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file)) as f:
                # Replace "1.jsnp" with "Snippet1" etc. to match file names in the CSV
                file_name = file.split(".")[0]
                file_name = f"Snippet{file_name}"
                code_snippets[file_name] = f.read()

        return code_snippets

    def _load_mean_scores(self, csv: str) -> dict:
        """
        Loads the mean scores from the CSV file.
        :param csv: Path to the CSV file containing the scores.
        :return: A pandas Series containing the mean scores.
        """
        data_frame = pd.read_csv(csv)

        # Drop the first column, which contains evaluator names
        data_frame = data_frame.drop(columns=data_frame.columns[0], axis=1)

        # Calculate the mean of the scores for each code snippet
        data_frame = data_frame.mean(axis=0)

        # Turn into dictionary with file names as keys and mean scores as values
        return data_frame.to_dict()

    def _tokenize_and_encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes and encodes the given text using the BERT tokenizer.
        :param text: The text to tokenize and encode.
        :return: A tuple containing the input_ids and the attention_mask.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        input_ids = tokenizer.encode(
            text, add_special_tokens=True, truncation=True, max_length=TOKEN_LENGTH
        )

        # Create an attention_mask
        attention_mask = [1] * len(input_ids) + [0] * (TOKEN_LENGTH - len(input_ids))

        # Ensure the input_ids have a maximum length of MAX_TOKEN_LENGTH
        if len(input_ids) < TOKEN_LENGTH:
            # Pad the input_ids with zeros to match MAX_TOKEN_LENGTH
            input_ids += [0] * (TOKEN_LENGTH - len(input_ids))
        else:
            # If the input_ids exceed MAX_TOKEN_LENGTH, truncate them
            # TODO: Necessary? Already done by tokenizer?
            input_ids = input_ids[:TOKEN_LENGTH]

        # Convert to PyTorch tensors
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        return input_ids, attention_mask


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
        batch_size: int = DEFAULT_BATCH_SIZE,
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
                scores = (
                    batch["scores"].unsqueeze(1).to(self.device)
                )  # Add dimension for matching batch size

                loss = self._train_iteration(
                    input_ids, scores, attention_mask=attention_mask
                )
                running_loss += loss

            print(
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
                scores = batch["scores"].to(self.device)

                y_batch.append(scores)
                predictions.append(self.model(input_ids, attention_mask))

            # Concatenate the lists of tensors to create a single tensor
            y_batch = torch.cat(y_batch, dim=0)
            predictions = torch.cat(predictions, dim=0)

            # Compute Mean Squared Error (MSE) using PyTorch
            mse = torch.mean((y_batch - predictions) ** 2).item()

            # Print the Mean Squared Error (MSE)
            print(f"Mean Squared Error (MSE): {mse}")


if __name__ == "__main__":
    data_dir = (
        "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/"
        # "Datasets/Dataset/Dataset_test/"
        "Datasets/Dataset/Dataset/"
    )
    snippets_dir = os.path.join(data_dir, "Snippets")
    csv = os.path.join(data_dir, "scores.csv")

    data_loader = CsvFolderDataLoader()
    train_loader, test_loader = data_loader.load(csv, snippets_dir)
    classifier = CodeReadabilityClassifier(train_loader, test_loader)
    classifier.train()
    classifier.evaluate()
