import pytest

from src.readability_classifier.keras.history_processing import HistoryProcessor


@pytest.fixture()
def processor():
    return HistoryProcessor()
