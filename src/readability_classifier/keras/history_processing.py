import logging
import pickle
from dataclasses import dataclass

import keras
import numpy as np

from src.readability_classifier.utils.utils import (
    calculate_auc,
    calculate_f1_score,
    calculate_mcc,
    calculate_precision,
    calculate_recall,
    get_from_dict,
)

# TODO: Add TrainStats and ValidationStats to the history


@dataclass
class HistoryList:
    """
    A list of histories.
    """

    fold_histories: list[keras.callbacks.History]

    def __getstate__(self):
        """
        Convert each History object to its state representation for serialization
        """
        return {"histories": [h.history for h in self.fold_histories]}

    def __setstate__(self, state):
        """
        Restore the object's state from the serialized state representation
        :param state: The state.
        :return: None
        """
        self.fold_histories = [keras.callbacks.History() for _ in state["histories"]]
        for h, history_state in zip(
            self.fold_histories, state["histories"], strict=False
        ):
            h.history = history_state


@dataclass
class Stats:
    """
    A class for storing the statistics of a model.
    """

    acc: float
    precision: float
    recall: float
    auc: float
    f1: float
    mcc: float


@dataclass
class TrainStats:
    """
    A class for storing the statistics of a train.
    """

    acc: float
    loss: float


@dataclass
class ValidationStats:
    """
    A class for storing the statistics of a validation.
    """

    acc: float
    loss: float


@dataclass
class EpochStats:
    """
    A class for storing the statistics of an epoch.
    """

    train_stats: TrainStats | None
    val_stats: ValidationStats | None
    tp: int
    tn: int
    fp: int
    fn: int
    acc: float
    precision: float
    recall: float
    auc: float
    f1: float
    mcc: float

    def to_stats(self) -> Stats:
        """
        Convert the epoch statistics to statistics.
        :return: The statistics.
        """
        return Stats(
            acc=self.acc,
            precision=self.precision,
            recall=self.recall,
            auc=self.auc,
            f1=self.f1,
            mcc=self.mcc,
        )


@dataclass
class FoldStats:
    """
    A class for storing the statistics of a fold.
    """

    fold_index: int
    epoch_stats: list[EpochStats]
    bst_epoch: Stats | None = None

    def __init__(
        self, fold_index: int, epoch_stats: list[EpochStats], metric: str = "acc"
    ):
        """
        Initialize the FoldStats class.
        :param fold_index: The index of the fold.
        :param epoch_stats: The statistics of the epochs.
        :param metric: The metric to use.
        """
        self.fold_index = fold_index
        self.epoch_stats = epoch_stats
        self.bst_epoch = self._best(metric=metric)

    def get_best(self):
        return self.bst_epoch

    def _best(self, metric: str = "acc") -> Stats:
        """
        Get the best epoch.
        :param metric: The metric to use.
        :return: The best epoch.
        """
        if metric == "acc":
            return max(self.epoch_stats, key=lambda x: x.acc).to_stats()
        if metric == "loss":
            return min(self.epoch_stats, key=lambda x: x.loss).to_stats()
        if metric == "auc":
            return max(self.epoch_stats, key=lambda x: x.auc).to_stats()
        if metric == "f1":
            return max(self.epoch_stats, key=lambda x: x.f1).to_stats()
        if metric == "mcc":
            return max(self.epoch_stats, key=lambda x: x.mcc).to_stats()

        raise ValueError(f"{metric} is not a valid metric.")


@dataclass
class OverallStats:
    """
    A class for storing the overall statistics.
    """

    fold_stats: list[FoldStats]
    best_fold: Stats | None = None
    average_fold: Stats | None = None

    def __init__(self, fold_stats: list[FoldStats], metric: str = "acc"):
        """
        Initialize the OverallStats class.
        :param fold_stats: The statistics of the folds.
        :param metric: The metric to use.
        """
        self.fold_stats = fold_stats
        self.best_fold = self._best(metric=metric)
        self.average_fold = self._average()

    def get_best(self):
        return self.best_fold

    def get_average(self):
        return self.average_fold

    def _best(self, metric: str = "acc") -> Stats:
        """
        Get the best fold.
        :param metric: The metric to use.
        :return: The statistics of the best fold.
        """
        if metric == "acc":
            return max(self.fold_stats, key=lambda x: x.get_best().acc).get_best()
        if metric == "loss":
            return min(self.fold_stats, key=lambda x: x.get_best().loss).get_best()
        if metric == "auc":
            return max(self.fold_stats, key=lambda x: x.get_best().auc).get_best()
        if metric == "f1":
            return max(self.fold_stats, key=lambda x: x.get_best().f1).get_best()
        if metric == "mcc":
            return max(self.fold_stats, key=lambda x: x.get_best().mcc).get_best()

        raise ValueError(f"{metric} is not a valid metric.")

    def _average(self) -> Stats:
        """
        Get the average over all folds.
        :return: The average over all folds.
        """
        best_folds = []
        for fold in self.fold_stats:
            best_folds.append(fold.get_best())

        return Stats(
            acc=np.mean([fold.acc for fold in best_folds]),
            precision=np.mean([fold.precision for fold in best_folds]),
            recall=np.mean([fold.recall for fold in best_folds]),
            auc=np.mean([fold.auc for fold in best_folds]),
            f1=np.mean([fold.f1 for fold in best_folds]),
            mcc=np.mean([fold.mcc for fold in best_folds]),
        )


class HistoryProcessor:
    """
    A class for processing the history of a model.
    """

    def evaluate(self, history: HistoryList) -> OverallStats:
        """
        Evaluate the model.
        :param history: The training history.
        :return: The overall statistics.s
        """
        fold_stats = []
        for epoch_time, fold_history in enumerate(history.fold_histories):
            fold_statum = self.evaluate_fold(
                fold_index=epoch_time, fold_history=fold_history
            )
            fold_stats.append(fold_statum)

        overall_stats = OverallStats(fold_stats=fold_stats, metric="acc")
        best_overall = overall_stats.get_best()
        average_overall = overall_stats.get_average()

        logging.info(
            "Overall results:\n"
            f"Best validation acc score: {best_overall.acc}\n"
            f"Best validation precision score: {best_overall.precision}\n"
            f"Best validation recall score: {best_overall.recall}\n"
            f"Best validation f1 score: {best_overall.f1}\n"
            f"Best validation auc score: {best_overall.auc}\n"
            f"Best validation mcc score: {best_overall.mcc}\n"
            f"Average validation acc score: {average_overall.acc}\n"
            f"Average validation precision score: {average_overall.precision}\n"
            f"Average validation recall score: {average_overall.recall}\n"
            f"Average validation f1 score: {average_overall.f1}\n"
            f"Average validation auc score: {average_overall.auc}\n"
        )

        return overall_stats

    def evaluate_fold(
        self,
        fold_index: int,
        fold_history: keras.callbacks.History,
    ) -> FoldStats:
        """
        Evaluate the model for a fold.
        :param fold_index: The index of the fold.
        :param fold_history: The history of the fold.
        :return: The training accuracy, f1, auc, and mcc of the fold.
        """
        history_dict = fold_history.history
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

        epoch_stats = []
        for i in range(len(val_false_negatives)):
            epoch_statum = self.evaluate_epoch(
                epoch_index=i,
                false_negatives=val_false_negatives,
                false_positives=val_false_positives,
                true_negatives=val_true_negatives,
                true_positives=val_true_positives,
            )
            epoch_stats.append(epoch_statum)

        fold_stats = FoldStats(
            epoch_stats=epoch_stats, fold_index=fold_index, metric="acc"
        )
        best_fold = fold_stats.get_best()

        logging.info(
            f"Fold {fold_index} results:\n"
            f"Best validation acc score: {best_fold.acc}\n"
            f"Best validation precision score: {best_fold.precision}\n"
            f"Best validation recall score: {best_fold.recall}\n"
            f"Best validation f1 score: {best_fold.f1}\n"
            f"Best validation auc score: {best_fold.auc}\n"
            f"Best validation mcc score: {best_fold.mcc}\n"
        )

        return fold_stats

    @staticmethod
    def evaluate_epoch(
        epoch_index: int,
        false_negatives: list[int],
        false_positives: list[int],
        true_negatives: list[int],
        true_positives: list[int],
    ) -> EpochStats:
        """
        Evaluate an epoch of the model.
        :param epoch_index: The index of the epoch.
        :param false_negatives: The list of false negatives.
        :param false_positives: The list of false positives.
        :param true_negatives: The list of true negatives.
        :param true_positives: The list of true positives.
        :return: The statistics of the epoch.
        """
        tp = true_positives[epoch_index]
        tn = true_negatives[epoch_index]
        fp = false_positives[epoch_index]
        fn = false_negatives[epoch_index]
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = calculate_mcc(tp=tp, tn=tn, fp=fp, fn=fn)
        precision = calculate_precision(tp=tp, fp=fp)
        recall = calculate_recall(tp=tp, fn=fn)
        f1 = calculate_f1_score(precision=precision, recall=recall)
        auc = calculate_auc(precision=precision, recall=recall)
        return EpochStats(
            train_stats=None,
            val_stats=None,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            acc=acc,
            precision=precision,
            recall=recall,
            auc=auc,
            f1=f1,
            mcc=mcc,
        )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("history_processing.log"),
            logging.StreamHandler(),
        ],
    )

    # Load a history
    with open("history.pkl", "rb") as file:
        loaded_history = pickle.load(file)

    HistoryProcessor().evaluate(loaded_history)
