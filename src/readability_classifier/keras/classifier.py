import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

import keras
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint

from src.readability_classifier.keras.history_processing import (
    HistoryList,
    HistoryProcessor,
)
from src.readability_classifier.keras.legacy_encoders import preprocess_data
from src.readability_classifier.keras.model import create_towards_model
from src.readability_classifier.models.encoders.dataset_utils import (
    Fold,
    ReadabilityDataset,
    split_k_fold,
)

# Define parameters
MODEL_OUTPUT = "../../res/keras/Experimental output/towards_best.h5"
STORE_DIR = "../../res/keras/Experimental output"
STATS_FILE_NAME = "stats.json"

# Seed
SEED = 42
random.seed(SEED)


def convert_to_towards_inputs(encoded_data: ReadabilityDataset) -> list[dict]:
    """
    Convert the encoded data to towards input.
    :param encoded_data: The encoded data.
    :return: The towards input.
    """
    return [
        {
            "structure": x["matrix"].numpy(),
            "image": np.transpose(x["image"], (1, 2, 0)).numpy(),
            "token": x["bert"]["input_ids"].numpy(),
            "segment": x["bert"]["segment_ids"].numpy()
            if "segment_ids" in x["bert"]
            else x["bert"]["position_ids"].numpy(),
            "label": x["score"].numpy(),
        }
        for x in encoded_data
    ]


class Classifier:
    """
    A source code readability classifier.
    """

    def __init__(
        self,
        model: keras.Model,
        encoded_data: ReadabilityDataset = None,
        k_fold: int = 10,
        epochs: int = 10,
        batch_size: int = 42,
    ):
        """
        Initializes the classifier.
        :param model: The model.
        :param encoded_data: The encoded data.
        :param k_fold: The number of folds.
        :param epochs: The number of epochs.
        :param batch_size: The batch size.
        """
        self.model = model
        self.encoded_data = encoded_data
        self.k_fold = k_fold
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self) -> HistoryList:
        """
        Train the model.
        :return: The training history.
        """
        towards_inputs = (
            convert_to_towards_inputs(self.encoded_data)
            if self.encoded_data is not None
            else preprocess_data()
        )

        random.shuffle(towards_inputs)

        logging.info(
            f"Number of samples: {len(towards_inputs)}\n"
            "Shapes:\n"
            f"Structure: {towards_inputs[0]['structure'].shape}\n"
            f"Image: {towards_inputs[0]['image'].shape}\n"
            f"Token: {towards_inputs[0]['token'].shape}\n"
            f"Segment: {towards_inputs[0]['segment'].shape}\n"
            "Label: 1\n"
        )

        history = HistoryList([])
        folds = split_k_fold(ReadabilityDataset(towards_inputs), k_fold=self.k_fold)
        for fold_index, fold in enumerate(folds):
            logging.info(f"Starting fold {fold_index + 1}/{self.k_fold}")
            fold_history = self.train_fold(fold)
            history.fold_histories.append(fold_history)

        return history

    def train_fold(self, fold: Fold) -> keras.callbacks.History:
        """
        Train the model for a fold.
        :param fold: The fold.
        :return: The history of the fold.
        """
        # Reset the model
        self.model.reset_states()

        # Train the model
        checkpoint = ModelCheckpoint(
            MODEL_OUTPUT, monitor="val_acc", verbose=1, save_best_only=True, model="max"
        )
        callbacks = [checkpoint]
        return self.model.fit(
            x=self._dataset_to_input(fold.train_set),
            y=self._dataset_to_label(fold.train_set),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            validation_data=(
                self._dataset_to_input(fold.val_set),
                self._dataset_to_label(fold.val_set),
            ),
        )

    @staticmethod
    def _dataset_to_input(
        dataset: ReadabilityDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a dataset to numpy arrays:
        structure_input, token_input, segment_input, image_input
        :param dataset: The dataset.
        :return: The input for the towards model.
        """
        structure = np.asarray([x["structure"] for x in dataset])
        image = np.asarray([x["image"] for x in dataset])
        token = np.asarray([x["token"] for x in dataset])
        segment = np.asarray([x["segment"] for x in dataset])
        return structure, token, segment, image

    @staticmethod
    def _dataset_to_label(dataset: ReadabilityDataset) -> np.ndarray:
        """
        Convert a dataset to towards output/score.
        :param dataset: The dataset.
        :return: The towards output.
        """
        return np.asarray([x["label"] for x in dataset])


def main():
    """
    Main function.
    :return: None
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("towards.log"),
            logging.StreamHandler(),
        ],
    )

    # Train the model
    towards_model = create_towards_model()
    classifier = Classifier(towards_model)
    history = classifier.train()

    # Save the history
    processed_history = HistoryProcessor().evaluate(history)
    store_path = Path(STORE_DIR) / STATS_FILE_NAME
    with open(store_path, "w") as file:
        json.dump(asdict(processed_history), file, indent=4)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    main()
