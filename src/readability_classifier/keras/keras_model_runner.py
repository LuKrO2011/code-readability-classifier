import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.readability_classifier.keras.classifier import Classifier
from src.readability_classifier.keras.history_processing import HistoryProcessor
from src.readability_classifier.keras.model import create_towards_model
from src.readability_classifier.model_runner import ModelRunnerInterface
from src.readability_classifier.models.encoders.dataset_utils import ReadabilityDataset

STATS_FILE_NAME = "stats.json"


class KerasModelRunner(ModelRunnerInterface):
    """
    A keras model runner. Runs the training, prediction and evaluation of a
    keras readability classifier.
    """

    def _run_without_cross_validation(
        self, parsed_args, encoded_data: ReadabilityDataset
    ):
        """
        Runs the training of the readability classifier without cross-validation.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        logging.warning("Keras only supports cross-validation.")
        self._run_with_cross_validation(parsed_args, encoded_data)

    # TODO: Use parsed args
    def _run_with_cross_validation(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the training of the readability classifier with cross-validation.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        # Get the parsed arguments
        # model = parsed_args.model
        store_dir = parsed_args.save
        # batch_size = parsed_args.batch_size
        # num_epochs = parsed_args.epochs
        # learning_rate = parsed_args.learning_rate

        # Build the model
        towards_model = create_towards_model()
        classifier = Classifier(towards_model, encoded_data)

        # Train the model
        history = classifier.train()
        processed_history = HistoryProcessor().evaluate(history)

        # Load the compare stats
        store_path = Path(store_dir) / STATS_FILE_NAME
        with open(store_path, "w") as file:
            json.dump(asdict(processed_history), file, indent=4)

    def run_predict(self, parsed_args):
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        raise NotImplementedError("Keras prediction is not implemented yet.")

    def run_evaluate(self, parsed_args):
        """
        Runs the evaluation of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        raise NotImplementedError("Keras evaluation is not implemented yet.")
