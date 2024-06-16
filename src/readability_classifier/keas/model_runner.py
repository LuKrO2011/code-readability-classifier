import json
import logging
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import keras.models

from src.readability_classifier.encoders.dataset_encoder import decode_score
from src.readability_classifier.encoders.dataset_utils import ReadabilityDataset
from src.readability_classifier.keas.classifier import Classifier
from src.readability_classifier.keas.history_processing import HistoryProcessor
from src.readability_classifier.keas.model import BertEmbedding, create_towards_model
from src.readability_classifier.toch.model_runner import ModelRunnerInterface

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

        # Log model summary
        logging.info(towards_model.summary(show_trainable=True))

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

        # Store the stats
        store_path = Path(store_dir) / STATS_FILE_NAME
        with open(store_path, "w") as file:
            json.dump(asdict(processed_history), file, indent=4)

    def run_predict(
        self, parsed_args, encoded_dataset: ReadabilityDataset
    ) -> tuple[str, float]:
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_dataset: A dataset of encoded data points.
        :return: The prediction as binary and as float (1 = readable, 0 = not readable).
        """
        model_path = parsed_args.model

        # TODO: Now requires which model to use -> Add parameter for "PREDICT" and resolve it here
        # Load the model
        model = create_towards_model()

        # Create the classifier
        classifier = Classifier(
            model=model,
            model_path=model_path,
            encoded_data=encoded_dataset,
            # TODO: Add an optional batch_size parameter for "PREDICT" and resolve it here
            #batch_size=batch_size,
        )

        # Predict the snippets
        predictions = classifier.predict()

        score_sums: dict[str, float] = {}
        score_counts: dict[str, int] = {}
        for i in range(len(predictions)):
            filename = encoded_dataset[i]['name']
            prediction = predictions[i].item()
            directory = os.path.dirname(filename)
            # Sum and count scores for each directory
            if score_sums.get(directory) is None:
                score_sums[directory] = 0.0
                score_counts[directory] = 0
            score_sums[directory] += prediction
            score_counts[directory] += 1
            prediction = decode_score(prediction)
            logging.info(f"Readability of file {filename}: {prediction}")

        overall_score_sum = 0
        for directory, score_sum in score_sums.items():
            overall_score_sum += score_sum
            avg = score_sum / score_counts[directory]
            prediction = decode_score(avg)
            logging.info(f"Readability of directory {directory}: {prediction}")
        # Overall average for return value
        avg = overall_score_sum / len(encoded_dataset)
        prediction = decode_score(avg)
        logging.info(f"Readability of whole input: {prediction}")
        return prediction

    def run_evaluate(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the evaluation of the readability classifier.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        model_path = parsed_args.load
        batch_size = parsed_args.batch_size
        store_dir = parsed_args.save

        # Load the model
        model = keras.models.load_model(
            model_path, custom_objects={"BertEmbedding": BertEmbedding}
        )

        # Create the classifier
        classifier = Classifier(
            model=model,
            encoded_data=encoded_data,
            batch_size=batch_size,
        )

        # Evaluate the model
        metrics = classifier.evaluate()

        # Store the history as pkl
        store_path = Path(store_dir) / "metrics.pkl"
        with open(store_path, "wb") as file:
            pickle.dump(metrics, file)

        processed_history = HistoryProcessor().evaluate_metrics(metrics)

        # Store the stats
        store_path = Path(store_dir) / STATS_FILE_NAME
        with open(store_path, "w") as file:
            json.dump(asdict(processed_history), file, indent=4)
