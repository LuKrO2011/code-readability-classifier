import json
import pickle
from dataclasses import asdict

import pytest

from readability_classifier.keras.history_processing import HistoryProcessor


@pytest.fixture()
def processor():
    return HistoryProcessor()


def test_process(processor):
    input_path = "res/keras/history/legacy_history.pkl"
    compare_path = "res/keras/history/processed_legacy_history.json"

    # Load the history
    with open(input_path, "rb") as file:
        loaded_history = pickle.load(file)

    # Process the history
    stats = HistoryProcessor().evaluate(loaded_history)
    stats = asdict(stats)

    # # Store the stats as json
    # with open(store_path, "w") as file:
    #     json.dump(stats, file, indent=4)

    # Load the compare stats
    with open(compare_path) as file:
        compare_stats = json.load(file)

    # Compare the stats
    assert stats == compare_stats
