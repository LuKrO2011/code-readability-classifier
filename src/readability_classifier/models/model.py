import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

TOKEN_LENGTH = 512
DEFAULT_BATCH_SIZE = 8


class ReadabilityDataset(Dataset):
    def __init__(self, data_dict: dict[str, tuple[list[int], list[int], float]]):
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
    def __init__(self, num_classes):
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

    def forward(self, input_ids, attention_mask):
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


class CodeReadabilityClassifier:
    def __init__(
        self, batch_size=DEFAULT_BATCH_SIZE, num_epochs=10, learning_rate=0.001
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = None
        self.test_loader = None

    def prepare_data(self, csv, data_dir):
        """
        Loads the data and prepares it for training and evaluation.
        :param csv: Path to the CSV file containing the scores.
        :param data_dir: Path to the directory containing the code snippets.
        :return: None
        """
        # Load data
        aggregated_scores, code_snippets = self.load_data(csv, data_dir)

        # Tokenize and encode code snippets
        embeddings = {}
        for name, snippet in code_snippets.items():
            input_ids, attention_mask = self.tokenize_and_encode(snippet)
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
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.setup_model()

    def load_data(self, csv, data_dir):
        mean_scores = self._load_mean_scores(csv)
        code_snippets = self._load_code_snippets(data_dir)

        return mean_scores, code_snippets

    def _load_code_snippets(self, data_dir):
        """
        Loads the code snippets from the files to a dictionary. The file names are used
        as keys and the code snippets as values.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The code snippets as a dictionary.
        """
        code_snippets = {}

        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file)) as f:
                # Replace "1.jsnp" with "Snippet1" etc. to match file names in the CSV
                file_name = file.split(".")[0]
                file_name = f"Snippet{file_name}"
                code_snippets[file_name] = f.read()

        return code_snippets

    def _load_mean_scores(self, csv):
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

    def tokenize_and_encode(self, text):
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

    def setup_model(self):
        self.model = CNNModel(1)  # Set number of classes to 1 for regression
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _train_iteration(self, x_batch, y_batch, attention_mask=None):
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

    classifier = CodeReadabilityClassifier()
    classifier.prepare_data(csv, snippets_dir)
    classifier.train()
    classifier.evaluate()
