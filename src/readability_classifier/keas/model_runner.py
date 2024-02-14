import logging

import keras.models
from src.readability_classifier.keas.classifier import \
    convert_to_towards_input_without_score
from readability_classifier.encoders.dataset_encoder import decode_score
from readability_classifier.encoders.dataset_utils import ReadabilityDataset
from readability_classifier.model_runner_interface import ModelRunnerInterface
from src.readability_classifier.keas.model import BertEmbedding

STATS_FILE_NAME = "stats.json"


class KerasModelRunner(ModelRunnerInterface):
    """
    A keras model runner. Runs the training, prediction and evaluation of a
    keras readability classifier.
    """

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
