import json
import logging
from dataclasses import asdict
from pathlib import Path

import keras.models

from readability_classifier.encoders.dataset_utils import ReadabilityDataset
from readability_classifier.toch.model_runner import ModelRunnerInterface
from src.readability_classifier.keas.classifier import Classifier
from src.readability_classifier.keas.history_processing import HistoryProcessor
from src.readability_classifier.keas.model import BertEmbedding, create_towards_model

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

    def _run_with_cross_validation(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the training of the readability classifier with cross-validation.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        # Get the parsed arguments
        store_dir = parsed_args.save
        batch_size = parsed_args.batch_size
        num_folds = parsed_args.k_fold
        epochs = parsed_args.epochs
        learning_rate = parsed_args.learning_rate
        fine_tune = parsed_args.fine_tune
        layer_names_to_freeze = parsed_args.freeze

        # Build the model
        towards_model = create_towards_model(learning_rate=learning_rate)

        # Load the pretrained model if available
        if fine_tune is not None:
            pretrained_model = keras.models.load_model(
                fine_tune, custom_objects={"BertEmbedding": BertEmbedding}
            )
            towards_model.set_weights(pretrained_model.get_weights())

        # Create the classifier
        classifier = Classifier(
            model=towards_model,
            encoded_data=encoded_data,
            store_dir=store_dir,
            batch_size=batch_size,
            k_fold=num_folds,
            epochs=epochs,
            layer_names_to_freeze=layer_names_to_freeze,
        )

        # Train the model
        history = classifier.train()
        processed_history = HistoryProcessor().evaluate(history)

        # Load the compare stats
        store_path = Path(store_dir) / STATS_FILE_NAME
        with open(store_path, "w") as file:
            json.dump(asdict(processed_history), file, indent=4)

    def _run_fine_tuning(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the fine-tuning of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        raise NotImplementedError("Keras fine-tuning is not implemented yet.")

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
