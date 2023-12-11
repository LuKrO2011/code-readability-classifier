import logging
import random
from pathlib import Path

import keras
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint

from readability_classifier.encoders.dataset_utils import (
    Fold,
    ReadabilityDataset,
    split_k_fold,
)
from src.readability_classifier.keas.history_processing import HistoryList

# Define parameters
STATS_FILE_NAME = "stats.json"
DEFAULT_STORE_DIR = "output"


# TODO : UserWarning: You are saving your model as an HDF5 file via `model.save()`.
#  This file format is considered legacy. We recommend using instead the native Keras
#  format, e.g. `model.save('my_model.keras')`. But: For checkpoint ".keras" is not
#  working


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
        epochs: int = 20,
        batch_size: int = 42,
        layer_names_to_freeze: list[str] = None,
        store_dir: str = DEFAULT_STORE_DIR,
    ):
        """
        Initializes the classifier.
        :param model: The model.
        :param encoded_data: The encoded data.
        :param k_fold: The number of folds.
        :param epochs: The number of epochs.
        :param batch_size: The batch size.
        :param layer_names_to_freeze: The layer names to freeze.
        :param store_dir: The store directory.
        """
        self.model = model
        self.initial_weights = model.get_weights()
        self.encoded_data = encoded_data
        self.k_fold = k_fold
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_names_to_freeze = layer_names_to_freeze or []
        self.store_dir = store_dir

    def train(self) -> HistoryList:
        """
        Train the model.
        :return: The training history.
        """
        towards_inputs = (
            convert_to_towards_inputs(self.encoded_data)
            if self.encoded_data is not None
            else None
        )

        if towards_inputs is None:
            raise NotImplementedError(
                "Keras training with old data encoders is not supported anymore."
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
            fold_history = self.train_fold(fold, fold_index + 1)
            history.fold_histories.append(fold_history)

        return history

    def train_fold(self, fold: Fold, fold_index: int = -1) -> keras.callbacks.History:
        """
        Train the model for a fold.
        :param fold: The fold.
        :param fold_index: The fold index.
        :return: The history of the fold.
        """
        # Reset the model
        model = self.model
        model.set_weights(self.initial_weights)

        # Freeze layers
        for layer in model.layers:
            if layer.name in self.layer_names_to_freeze:
                layer.trainable = False

        # Train the model
        store_path = str(Path(self.store_dir) / f"model_{fold_index}.h5")
        checkpoint = ModelCheckpoint(
            store_path, monitor="val_acc", verbose=1, save_best_only=True, model="max"
        )
        callbacks = [checkpoint]
        return model.fit(
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
