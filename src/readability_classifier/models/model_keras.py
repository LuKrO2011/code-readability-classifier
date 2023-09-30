import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import BertTokenizer, DataCollatorWithPadding, TFBertModel

# WARNING: THIS FILE IS DEPRECATED!!!

# TODO: Make token length configurable
TOKEN_LENGTH = 512  # Maximum length of tokens for BERT
DEFAULT_BATCH_SIZE = 8  # Small to avoid memory errors


class CodeReadabilityRegressor(tf.keras.Model):
    """
    A regression model that predicts the readability of a code snippet. The model
    uses BERT as a feature extractor and a simple feed-forward network as a
    regressor.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialize the model.
        :param num_classes:
        """
        super().__init__()

        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_classes, activation="linear")

    def call(self, inputs: dict, **kwargs) -> tf.Tensor:
        """
        Call the model. This is the forward pass.
        :param inputs: The input data
        :param kwargs: Additional arguments
        :return: The output of the model
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.pooling(bert_output)
        x = self.fc1(pooled_output)
        return self.fc2(x)


class CsvFolderDataLoader:
    """
    A data loader for loading data from a CSV file and the corresponding code snippets.
    TODO: Add hierarchy for different dataset formats?
    TODO: Convert to huggingface datasets?
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initializes the data loader.
        :param batch_size: The batch size.
        """
        self.batch_size = batch_size

    def load(self, csv: str, data_dir: str) -> tuple[Dataset, Dataset]:
        """
        Loads the data and prepares it for training and evaluation.
        :param csv: Path to the CSV file containing the scores.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The training and test data.
        """
        aggregated_scores, code_snippets = self._load_from_storage(csv, data_dir)

        embeddings = []
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        for name, snippet in code_snippets.items():
            input_ids, attention_mask = self.tokenize_and_encode(snippet, tokenizer)
            embeddings.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "scores": aggregated_scores[name],
                }
            )

        # Convert to dataset
        dataset = Dataset.from_list(embeddings)
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "scores"]
        )

        # Split into train and test
        dataset = dataset.train_test_split(test_size=0.2)
        train_data = dataset["train"]
        test_data = dataset["test"]

        # Use hugging-faces to_tf_dataset to create a tf.data.Dataset
        # TODO: Why DataCollatorWithPadding?
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, return_tensors="tf"
        )
        train_data = train_data.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            label_cols=["scores"],
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )

        test_data = test_data.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            label_cols=["scores"],
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )

        return train_data, test_data

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

    @staticmethod
    def tokenize_and_encode(
        text: str, tokenizer: BertTokenizer
    ) -> tuple[list[int], list[int]]:
        """
        Tokenizes and encodes the given text using the given tokenizer.
        :param text: The text to tokenize and encode.
        :param tokenizer: The tokenizer to use.
        :return: A tuple containing the input IDs and the attention mask.
        """
        encoded_dict = tokenizer(
            text, add_special_tokens=True, truncation=True, max_length=TOKEN_LENGTH
        )

        # Pad the sequence to max length
        padded = tokenizer.pad(
            encoded_dict, padding="max_length", max_length=TOKEN_LENGTH
        )
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]
        # TODO: is padded["token_type_ids"] useful somehow?
        # token_type_ids = padded["token_type_ids"]

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
        train_generator=None,
        test_generator=None,
        model_path=None,
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=10,
        learning_rate=0.001,
    ):
        """
        Initializes the classifier.
        :param train_generator: The training data.
        :param test_generator: The test data.
        :param batch_size: The batch size.
        :param num_epochs: The number of epochs.
        :param learning_rate: The learning rate.
        """
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Set up the model on initialization
        self._setup_model()

    def _setup_model(self) -> None:
        """
        Sets up the model. This includes initializing the model, the loss function and
        the optimizer.
        :return: None
        """
        self.model = CodeReadabilityRegressor(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train(self) -> None:
        """
        Trains the model.
        :return: None
        """
        if self.train_generator is None:
            raise ValueError("No training data provided.")

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        return self.model.fit(
            x=self.train_generator,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=1,
        )

    def evaluate(self) -> None:
        """
        Evaluates the model.
        :return: None
        :return:
        """
        if self.test_generator is None:
            raise ValueError("No test data provided.")

        loss = self.model.evaluate(self.test_generator, verbose=0)
        logging.info(f"Test loss: {loss}")

    def store(self, path: str) -> None:
        """
        Stores the model at the given path.
        :param path: The path to store the model.
        :return: None
        """
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        """
        Loads the model from the given path.
        :param path: The path to load the model from.
        :return: None
        """
        self.model.load_weights(path)

    def predict(self, code_snippet: str) -> float:
        """
        Predicts the readability of the given code snippet.
        :param code_snippet: The code snippet to predict the readability of.
        :return: The predicted readability.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        input_ids, attention_mask = CsvFolderDataLoader.tokenize_and_encode(
            code_snippet, tokenizer
        )
        input_ids = np.array([input_ids])
        attention_mask = np.array([attention_mask])
        prediction = self.model.predict(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return prediction[0][0]


DATA_DIR = (
    "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Dataset/Dataset/"
)

if __name__ == "__main__":
    snippets_dir = os.path.join(DATA_DIR, "Snippets")
    csv_path = os.path.join(DATA_DIR, "scores.csv")

    # Load the data
    data_loader = CsvFolderDataLoader()
    train_loader, test_loader = data_loader.load(csv_path, snippets_dir)

    # Train and evaluate the model
    classifier = CodeReadabilityClassifier(train_loader, test_loader)
    classifier.train()
    classifier.evaluate()

    # Store the model
    classifier.store("model.pt")
