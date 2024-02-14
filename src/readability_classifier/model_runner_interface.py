from abc import ABC, abstractmethod

from readability_classifier.encoders.dataset_utils import (
    ReadabilityDataset,
)


class ModelRunnerInterface(ABC):
    """
    Interface for model runners.
    """

    @abstractmethod
    def run_predict(self, parsed_args, encoded_data: ReadabilityDataset) -> any:
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: A single encoded data point.
        :return: None
        """
        pass
