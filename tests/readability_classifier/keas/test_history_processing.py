import json
import pickle
from dataclasses import asdict
from pathlib import Path

import keras

from src.readability_classifier.keas.history_processing import (
    HistoryList,
    HistoryProcessor,
)
from tests.readability_classifier.utils.utils import HISTORY_FILE, STATS_FILE, DirTest


class TestHistoryProcessor(DirTest):
    def test_evaluate_epoch(self):
        train_loss = [0.1, 0.2, 0.3]
        train_acc = [0.9, 0.8, 0.7]
        val_loss = [0.4, 0.5, 0.6]
        val_acc = [0.6, 0.5, 0.4]
        false_negatives = [0, 1, 2]
        false_positives = [1, 2, 3]
        true_negatives = [3, 2, 1]
        true_positives = [2, 3, 4]

        epoch_stats = HistoryProcessor.evaluate_epoch(
            epoch_index=1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            false_negatives=false_negatives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            true_positives=true_positives,
        )

        assert epoch_stats.epoch == 1
        assert epoch_stats.train_stats.loss == 0.2
        assert epoch_stats.train_stats.acc == 0.8
        assert epoch_stats.val_stats.loss == 0.5
        assert epoch_stats.val_stats.acc == 0.5
        assert epoch_stats.acc == 0.625
        assert epoch_stats.precision == 0.6
        assert epoch_stats.recall == 0.75
        assert epoch_stats.auc == 0.675
        assert epoch_stats.f1 == 0.6666666666666665
        assert epoch_stats.mcc == 0.2581988897471611

    def test_evaluate_fold(self):
        fake_history = keras.callbacks.History()
        fake_history.history["loss"] = [0.1, 0.2, 0.3]
        fake_history.history["acc"] = [0.9, 0.8, 0.7]
        fake_history.history["val_loss"] = [0.1, 0.2, 0.3]
        fake_history.history["val_acc"] = [0.9, 0.8, 0.7]
        fake_history.history["val_false_negatives"] = [0, 1, 2]
        fake_history.history["val_false_positives"] = [1, 2, 3]
        fake_history.history["val_true_positives"] = [3, 2, 1]
        fake_history.history["val_true_negatives"] = [2, 3, 4]

        history_list = HistoryList([fake_history])

        fold_stats = HistoryProcessor().evaluate_fold(
            fold_index=0, fold_history=history_list.fold_histories[0]
        )

        assert len(fold_stats.epoch_stats) == 3

        # The stats of the best epoch are different, because the best epoch is not
        # necessarily the last epoch.
        best_epoch = fold_stats.best_epoch
        assert best_epoch.acc == 0.8333333333333334
        assert best_epoch.precision == 0.75
        assert best_epoch.recall == 1.0
        assert best_epoch.auc == 0.875
        assert best_epoch.f1 == 0.8571428571428571
        assert best_epoch.mcc == 0.7071067811865476

    def test_evaluate(self):
        fake_history = keras.callbacks.History()
        fake_history.history["loss"] = [0.1, 0.2, 0.3]
        fake_history.history["acc"] = [0.9, 0.8, 0.7]
        fake_history.history["val_loss"] = [0.1, 0.2, 0.3]
        fake_history.history["val_acc"] = [0.9, 0.8, 0.7]
        fake_history.history["val_false_negatives"] = [0, 1, 2]
        fake_history.history["val_false_positives"] = [1, 2, 3]
        fake_history.history["val_true_positives"] = [3, 2, 1]
        fake_history.history["val_true_negatives"] = [2, 3, 4]

        history_list = HistoryList([fake_history])

        overall_stats = HistoryProcessor().evaluate(history_list)

        assert len(overall_stats.fold_stats) == 1
        best_fold = overall_stats.best_fold
        average_fold = overall_stats.average_fold
        assert best_fold.acc == average_fold.acc == 0.8333333333333334
        assert best_fold.precision == average_fold.precision == 0.75
        assert best_fold.recall == average_fold.recall == 1.0
        assert best_fold.auc == average_fold.auc == 0.875
        assert best_fold.f1 == average_fold.f1 == 0.8571428571428571
        assert best_fold.mcc == average_fold.mcc == 0.7071067811865476

    def test_evaluate_from_pkl(self):
        with open(HISTORY_FILE, "rb") as file:
            history = pickle.load(file)

        actual_stats = HistoryProcessor().evaluate(history)

        # Convert to json
        file_name = "stats.json"
        path = Path(self.output_dir) / file_name
        with open(path, "w") as file:
            json.dump(asdict(actual_stats), file, indent=4)
        with open(path) as file:
            actual_stats = json.load(file)

        with open(STATS_FILE) as file:
            expected_stats = json.load(file)

        assert actual_stats == expected_stats
