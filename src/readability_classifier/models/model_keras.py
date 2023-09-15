import os

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import BertTokenizer, DataCollatorWithPadding, TFBertModel

TOKEN_LENGTH = 512  # Maximum length of tokens for BERT
DEFAULT_BATCH_SIZE = 8  # Small to avoid memory errors


# TODO: Use this for bert: https://huggingface.co/docs/datasets/use_dataset


class ReadabilityDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dict, batch_size=DEFAULT_BATCH_SIZE):
        self.data_dict = data_dict
        self.names = list(data_dict.keys())
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.names) / self.batch_size))

    def __getitem__(self, idx):
        batch_names = self.names[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = [self.data_dict[name] for name in batch_names]

        input_ids = np.array([item[0] for item in batch_data])
        attention_mask = np.array([item[1] for item in batch_data])
        scores = np.array([item[2] for item in batch_data])

        return ({"input_ids": input_ids, "attention_mask": attention_mask}, scores)


class CodeReadabilityRegressor(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_classes, activation="linear")

    def call(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.pooling(bert_output)
        x = self.fc1(pooled_output)
        return self.fc2(x)


class CsvFolderDataLoader:
    def __init__(self, batch_size=DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size

    def load(self, csv, data_dir):
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
        train_data.to_tf_dataset(
            columns=["input_ids", "attention_mask", "scores"],
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )

        test_data.to_tf_dataset(
            columns=["input_ids", "attention_mask", "scores"],
            batch_size=self.batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )

        return train_data, test_data

    def _load_from_storage(self, csv, data_dir):
        mean_scores = self._load_mean_scores(csv)
        code_snippets = self._load_code_snippets(data_dir)
        return mean_scores, code_snippets

    def _load_code_snippets(self, data_dir):
        code_snippets = {}
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file)) as f:
                file_name = file.split(".")[0]
                file_name = f"Snippet{file_name}"
                code_snippets[file_name] = f.read()
        return code_snippets

    def _load_mean_scores(self, csv):
        data_frame = pd.read_csv(csv)
        data_frame = data_frame.drop(columns=data_frame.columns[0], axis=1)
        data_frame = data_frame.mean(axis=0)
        return data_frame.to_dict()

    @staticmethod
    def tokenize_and_encode(text, tokenizer):
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
    def __init__(
        self,
        train_generator=None,
        test_generator=None,
        model_path=None,
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=10,
        learning_rate=0.001,
    ):
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self._setup_model()

    def _setup_model(self):
        self.model = CodeReadabilityRegressor(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train(self):
        if self.train_generator is None:
            raise ValueError("No training data provided.")

        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        return self.model.fit(
            x=self.train_generator,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=1,
        )

    def evaluate(self):
        if self.test_generator is None:
            raise ValueError("No test data provided.")

        loss = self.model.evaluate(self.test_generator, verbose=0)
        print("Mean Squared Error (MSE):", loss)

    def store(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, code_snippet):
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
