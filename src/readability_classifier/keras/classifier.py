import random

import keras
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint

from readability_classifier.keras.legacy_encoders import preprocess_data
from readability_classifier.keras.model import create_towards_model
from readability_classifier.models.encoders.dataset_utils import (
    Fold,
    ReadabilityDataset,
    split_k_fold,
)
from readability_classifier.utils.utils import (
    calculate_f1_score,
    calculate_mcc,
    calculate_precision,
    calculate_recall,
    get_from_dict,
)

# Define parameters
K_FOLD = 10
EPOCHS = 20
MODEL_OUTPUT = "../../res/keras/Experimental output/towards_best.h5"

# Seed
SEED = 42
random.seed(SEED)


def get_training_set(
    fold_index: int,
    num_sample: int,
    train_structure: np.ndarray,
    train_token: np.ndarray,
    train_segment: np.ndarray,
    train_image: np.ndarray,
    train_label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the training set.
    :param fold_index: The index of the fold.
    :param num_sample: The number of samples.
    :param train_structure: The training structure data.
    :param train_token: The training token data.
    :param train_segment: The training segment data.
    :param train_image: The training image data.
    :param train_label: The training label data.
    :return: The training set.
    """
    train_indices_1 = slice(0, fold_index * num_sample)
    train_indices_2 = slice((fold_index + 1) * num_sample, None)

    x_train_structure = np.concatenate(
        [train_structure[train_indices_1], train_structure[train_indices_2]], axis=0
    )
    x_train_token = np.concatenate(
        [train_token[train_indices_1], train_token[train_indices_2]], axis=0
    )
    x_train_segment = np.concatenate(
        [train_segment[train_indices_1], train_segment[train_indices_2]], axis=0
    )
    x_train_image = np.concatenate(
        [train_image[train_indices_1], train_image[train_indices_2]], axis=0
    )
    y_train = np.concatenate(
        [train_label[train_indices_1], train_label[train_indices_2]], axis=0
    )

    return x_train_structure, x_train_token, x_train_segment, x_train_image, y_train


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


# TODO: Remove computation of those: data_position and data_image (unused)
class Classifier:
    """
    A source code readability classifier.
    """

    label: np.ndarray = None
    structure: np.ndarray = None
    image: np.ndarray = None
    token: np.ndarray = None
    segment: np.ndarray = None

    def __init__(
        self, towards_model: keras.Model, encoded_data: ReadabilityDataset = None
    ):
        """
        Initializes the classifier.
        :param towards_model: The towards model.
        """
        self.towards_model = towards_model
        self.encoded_data = encoded_data

    def train(self) -> list[keras.callbacks.History]:
        """
        Train the model.
        :return: The training history.
        """
        towards_inputs = (
            convert_to_towards_inputs(self.encoded_data)
            if self.encoded_data is not None
            else preprocess_data()
        )

        # Shuffle the data
        random.shuffle(towards_inputs)

        # Extract the data from the towards inputs
        self.label = np.asarray([x["label"] for x in towards_inputs])
        self.structure = np.asarray([x["structure"] for x in towards_inputs])
        self.image = np.asarray([x["image"] for x in towards_inputs])
        self.token = np.asarray([x["token"] for x in towards_inputs])
        self.segment = np.asarray([x["segment"] for x in towards_inputs])

        print("Shape of structure data tensor:", self.structure.shape)
        print("Shape of image data tensor:", self.image.shape)
        print("Shape of token tensor:", self.token.shape)
        print("Shape of segment tensor:", self.segment.shape)
        print("Shape of label tensor:", self.label.shape)

        history = []
        folds = split_k_fold(ReadabilityDataset(towards_inputs), k_fold=K_FOLD)
        for fold_index, fold in enumerate(folds):
            print(f"Now is fold {fold_index}")
            fold_history = self.train_fold(fold)
            history.append(fold_history)

        return history

    def evaluate(self, history: list) -> None:
        """
        Evaluate the model.
        :param history: The training history.
        :return: None
        """
        f1s = []
        aucs = []
        mccs = []
        accs = []
        fold_index = 1
        for history_item in history:
            (
                train_acc,
                auc,
                f1,
                mcc,
            ) = self.evaluate_fold(
                epoch_time=fold_index,
                fold_history=history_item,
            )
            accs.append(train_acc)
            aucs.append(auc)
            f1s.append(f1)
            mccs.append(mcc)
            fold_index += 1
        print("Average training acc score", np.mean(accs))
        print("Average f1 score", np.mean(f1s))
        print("Average auc score", np.mean(aucs))
        print("Average mcc score", np.mean(mccs))
        print()

    def evaluate_fold(
        self,
        epoch_time: int,
        fold_history: keras.callbacks.History,
    ) -> tuple[float, float, float, float]:
        """
        Evaluate the model for a fold.
        :param epoch_time: The epoch time.
        :param fold_history: The history of the fold.
        :return: The training accuracy, f1, auc, and mcc of the fold.
        """
        history_dict = fold_history.history
        val_acc_values = get_from_dict(history_dict, "val_acc")
        val_recall_value = get_from_dict(history_dict, "val_recall")
        val_precision_value = get_from_dict(history_dict, "val_precision")
        val_auc_value = get_from_dict(history_dict, "val_auc")
        val_false_negatives = [
            int(x) for x in get_from_dict(history_dict, "val_false_negatives")
        ]
        val_false_positives = [
            int(x) for x in get_from_dict(history_dict, "val_false_positives")
        ]
        val_true_positives = [
            int(x) for x in get_from_dict(history_dict, "val_true_positives")
        ]
        val_true_negatives = [
            int(x) for x in get_from_dict(history_dict, "val_true_negatives")
        ]

        mccs = []
        f1s = []
        for i in range(EPOCHS):
            f1, mcc = self.evaluate_epoch(
                epoch_index=i,
                false_negatives=val_false_negatives,
                false_positives=val_false_positives,
                true_negatives=val_true_negatives,
                true_positives=val_true_positives,
            )
            f1s.append(f1)
            mccs.append(mcc)

        best_train_acc = np.max(val_acc_values)
        best_auc = np.max(val_auc_value)
        best_f1 = np.max(f1s)
        best_mcc = np.max(mccs)

        print("Processing fold #", epoch_time)
        print("------------------------------------------------")
        print("best accuracy score is #", np.max(val_acc_values))
        print("average recall score is #", np.mean(val_recall_value))
        print("average precision score is #", np.mean(val_precision_value))
        print("best f1 score is #", np.max(f1s))
        print("best auc score is #", np.max(val_auc_value))
        print("best mcc score is #", np.max(mccs))
        print()

        return (
            best_train_acc,
            best_auc,
            best_f1,
            best_mcc,
        )

    @staticmethod
    def evaluate_epoch(
        epoch_index: int,
        false_negatives: list[int],
        false_positives: list[int],
        true_negatives: list[int],
        true_positives: list[int],
    ) -> tuple[float, float]:
        """
        Evaluate an epoch of the model.
        :param epoch_index: The index of the epoch.
        :param false_negatives: The list of false negatives.
        :param false_positives: The list of false positives.
        :param true_negatives: The list of true negatives.
        :param true_positives: The list of true positives.
        :return: The f1 and mcc of the epoch.
        """
        tp = true_positives[epoch_index]
        tn = true_negatives[epoch_index]
        fp = false_positives[epoch_index]
        fn = false_negatives[epoch_index]
        mcc = calculate_mcc(tp=tp, tn=tn, fp=fp, fn=fn)
        precision = calculate_precision(tp=tp, fp=fp)
        recall = calculate_recall(tp=tp, fn=fn)
        f1 = calculate_f1_score(precision=precision, recall=recall)
        return f1, mcc

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
    towards_model = create_towards_model()
    classifier = Classifier(towards_model)
    history = classifier.train()
    classifier.evaluate(history)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    main()
