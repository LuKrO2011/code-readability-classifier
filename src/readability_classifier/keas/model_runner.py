import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import keras.models

from readability_classifier.encoders.dataset_encoder import decode_score
from readability_classifier.encoders.dataset_utils import ReadabilityDataset
from readability_classifier.toch.model_runner import ModelRunnerInterface
from src.readability_classifier.keas.classifier import (
    Classifier,
    convert_to_towards_input_without_score,
)
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

        # Freeze the input layers
        for layer in towards_model.layers:
            if layer.name in layer_names_to_freeze:
                layer.trainable = False

        # Create the classifier
        classifier = Classifier(
            model=towards_model,
            encoded_data=encoded_data,
            store_dir=store_dir,
            batch_size=batch_size,
            k_fold=num_folds,
            epochs=epochs,
        )

        # Train the model
        history = classifier.train()

        # Store the history as pkl
        store_path = Path(store_dir) / "history.pkl"
        with open(store_path, "wb") as file:
            pickle.dump(history, file)

        processed_history = HistoryProcessor().evaluate(history)

        # Load the compare stats
        store_path = Path(store_dir) / STATS_FILE_NAME
        with open(store_path, "w") as file:
            json.dump(asdict(processed_history), file, indent=4)

    def run_predict(
        self, parsed_args, encoded_data: ReadabilityDataset
    ) -> tuple[str, float]:
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: A single encoded data point.
        :return: The prediction as binary and as float (1 = readable, 0 = not readable).
        """
        model_path = parsed_args.model

        # Load the model
        model = keras.models.load_model(
            model_path, custom_objects={"BertEmbedding": BertEmbedding}
        )

        # Predict the readability of the snippet
        towards_input = convert_to_towards_input_without_score(encoded_data)
        prediction = model.predict(towards_input)
        prediction = decode_score(prediction)
        logging.info(f"Readability of snippet: {prediction}")
        return prediction

    def run_evaluate(self, parsed_args):
        """
        Runs the evaluation of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        raise NotImplementedError(
            "Keras evaluation is not implemented, as the model is evaluated during"
            "training."
        )
