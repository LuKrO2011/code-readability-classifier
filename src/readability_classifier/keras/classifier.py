import logging
import pickle
import random

import keras
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint

from readability_classifier.keras.history_processing import (
    HistoryList,
    HistoryProcessor,
)
from readability_classifier.keras.legacy_encoders import preprocess_data
from readability_classifier.keras.model import create_towards_model
from readability_classifier.models.encoders.dataset_utils import (
    Fold,
    ReadabilityDataset,
    split_k_fold,
)

# Define parameters
K_FOLD = 10
EPOCHS = 20
MODEL_OUTPUT = "../../res/keras/Experimental output/towards_best.h5"

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
            "structure": x["matrix"],
            "image": np.transpose(x["image"], (1, 2, 0)),
            "token": x["bert"]["input_ids"],
            "segment": x["bert"]["segment_ids"]
            if "segment_ids" in x["bert"]
            else x["bert"]["token_type_ids"],
            "label": x["score"],
        }
        for x in encoded_data
    ]


class Classifier:
    """
    A source code readability classifier.
    """

    def __init__(
        self, towards_model: keras.Model, encoded_data: ReadabilityDataset = None
    ):
        """
        Initializes the classifier.
        :param towards_model: The towards model.
        """
        self.towards_model = towards_model
        self.encoded_data = encoded_data

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
        folds = split_k_fold(ReadabilityDataset(towards_inputs), k_fold=K_FOLD)
        for fold_index, fold in enumerate(folds):
            logging.info(f"Starting fold {fold_index + 1}/{K_FOLD}")
            fold_history = self.train_fold(fold)
            history.fold_histories.append(fold_history)

        return history

    def train_fold(self, fold: Fold) -> keras.callbacks.History:
        """
        Train the model for a fold.
        :param fold: The fold.
        :return: The history of the fold.
        """
        # Train the model
        towards_model = create_towards_model()
        checkpoint = ModelCheckpoint(
            MODEL_OUTPUT, monitor="val_acc", verbose=1, save_best_only=True, model="max"
        )
        callbacks = [checkpoint]
        return towards_model.fit(
            x=self._dataset_to_input(fold.train_set),
            y=self._dataset_to_label(fold.train_set),
            epochs=EPOCHS,
            batch_size=42,
            callbacks=callbacks,
            verbose=0,
            validation_data=(
                self._dataset_to_input(fold.val_set),
                self._dataset_to_label(fold.val_set),
            ),
        )

    def _dataset_to_input(
        self, dataset: ReadabilityDataset
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

    def _dataset_to_label(self, dataset: ReadabilityDataset) -> np.ndarray:
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

    # Store the history
    with open("history.pkl", "wb") as file:
        pickle.dump(history, file)

    # Evaluate the model
    HistoryProcessor().evaluate(history)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    main()
