from abc import ABC, abstractmethod

from readability_classifier.encoders.dataset_utils import (
    ReadabilityDataset,
)


class ModelRunnerInterface(ABC):
    """
    Interface for model runners.
    """

    def run_train(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the training of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        k_fold = parsed_args.k_fold
        if k_fold == 0:
            self._run_without_cross_validation(parsed_args, encoded_data)
        else:
            self._run_with_cross_validation(parsed_args, encoded_data)

    @abstractmethod
    def _run_without_cross_validation(
        self, parsed_args, encoded_data: ReadabilityDataset
    ):
        """
        Runs the training of the readability classifier without cross-validation.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        pass

    @abstractmethod
    def _run_with_cross_validation(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the training of the readability classifier with cross-validation.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        pass

    @abstractmethod
    def run_predict(self, parsed_args, encoded_data: ReadabilityDataset) -> any:
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: A single encoded data point.
        :return: None
        """
        pass

    @abstractmethod
    def run_evaluate(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the evaluation of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        pass
