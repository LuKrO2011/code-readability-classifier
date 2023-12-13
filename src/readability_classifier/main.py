import logging
import os
import random
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any

from readability_classifier.encoders.dataset_encoder import DatasetEncoder
from readability_classifier.encoders.dataset_utils import (
    load_encoded_dataset,
    load_raw_dataset,
    store_encoded_dataset,
)
from readability_classifier.toch.model_buider import Model
from readability_classifier.toch.model_runner import (
    ModelRunnerInterface,
    TorchModelRunner,
)
from src.readability_classifier.keas.model_runner import KerasModelRunner

DEFAULT_LOG_FILE_NAME = "readability-classifier"
DEFAULT_LOG_FILE = f"{DEFAULT_LOG_FILE_NAME}.log"
DEFAULT_MODEL_FILE = "model"
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SAVE_DIR = os.path.join(CURR_DIR, "../../models")
KERAS = True
SEED = 42


def _setup_logging(log_file: str = DEFAULT_LOG_FILE, overwrite: bool = False) -> None:
    """
    Set up logging.
    """
    # Create the save dir and file if they do not exist
    if not os.path.isdir(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    if not os.path.isfile(log_file):
        with open(log_file, "w") as _:
            pass

    # Get the overwrite flag
    mode = "w" if overwrite else "a"

    # Set the logging level
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)

    # Create a file handler to write messages to a log file
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging_level)

    # Create a console handler to display messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)

    # Define the log format
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Get the root logger and add the handlers
    logger = logging.getLogger("")
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class TaskNotSupportedException(Exception):
    """
    Exception is thrown whenever a task is not supported.
    """


class Tasks(Enum):
    """
    Enum for the different tasks of the readability classifier.
    """

    ENCODE = "ENCODE"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    PREDICT = "PREDICT"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        return self.value


def _set_up_arg_parser() -> ArgumentParser:
    """
    Parses the arguments for the readability classifier.
    :return_:   Returns the parser for extracting the arguments.
    """
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)

    # Parser for the encoding task
    encode_parser = sub_parser.add_parser(str(Tasks.ENCODE))
    encode_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the dataset.",
    )
    encode_parser.add_argument(
        "--intermediate",
        required=False,
        type=Path,
        help="Path to where the encoded dataset should be stored. ",
    )
    encode_parser.add_argument(
        "--save",
        "-s",
        required=False,
        type=Path,
        help="Path where the log file should be stored. If not specified, "
        "the log file is stored in the current directory.",
        default=DEFAULT_SAVE_DIR,
    )

    # Parser for the training task
    train_parser = sub_parser.add_parser(str(Tasks.TRAIN))
    train_parser.add_argument(
        "--model",
        "-m",
        required=False,
        type=Model,
        help="The model to use.",
        default=Model.TOWARDS,
    )
    train_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the dataset.",
    )
    train_parser.add_argument(
        "--encoded",
        required=False,
        default=False,
        action="store_true",
        help="Set this flag if the dataset is already encoded.",
    )
    train_parser.add_argument(
        "--save",
        "-s",
        required=False,
        type=Path,
        help="Path to the folder where the model should be stored. If not specified, "
        "the model is not stored.",
        default=DEFAULT_SAVE_DIR,
    )
    train_parser.add_argument(
        "--intermediate",
        required=False,
        type=Path,
        help="Path to the folder where intermediate results should be stored. "
        "If not specified, the intermediate results are not stored.",
    )
    train_parser.add_argument(
        "--evaluate",
        required=False,
        default=True,
        action="store_false",
        help="Whether the model should be evaluated after training.",
    )
    train_parser.add_argument(
        "--k-fold",
        "-k",
        required=False,
        type=int,
        default=10,
        help="The number of folds for k-fold cross-validation.",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        required=False,
        type=int,
        default=8,
        help="The batch size for training.",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        required=False,
        type=int,
        default=20,
        help="The number of epochs for training.",
    )
    train_parser.add_argument(
        "--learning-rate",
        "-r",
        required=False,
        type=float,
        default=0.0015,
        help="The learning rate for training.",
    )
    train_parser.add_argument(
        "--fine-tune",
        required=False,
        default=None,
        type=Path,
        help="Path to the model to fine-tune. If not specified, no fine-tuning is "
        "performed and the model is trained from scratch.",
    )
    train_parser.add_argument(
        "--freeze",
        required=False,
        default=[],
        nargs="+",
        type=str,
        help="The layer names to freeze.",
    )

    # Parser for the evaluation task
    evaluate_parser = sub_parser.add_parser(str(Tasks.EVALUATE))
    evaluate_parser.add_argument(
        "--load",
        "-l",
        required=True,
        type=Path,
        help="Path to the model to load for evaluation.",
    )
    evaluate_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the validation dataset.",
    )
    evaluate_parser.add_argument(
        "--encoded",
        required=False,
        default=False,
        action="store_true",
        help="Set this flag if the dataset is already encoded.",
    )
    evaluate_parser.add_argument(
        "--save",
        "-s",
        required=False,
        type=Path,
        help="Path to the folder where the evaluation results should be stored. "
        "If not specified, the results are not stored.",
    )
    evaluate_parser.add_argument(
        "--model",
        "-m",
        required=False,
        type=Model,
        help="The type of the model used.",
        default=Model.TOWARDS,
    )
    evaluate_parser.add_argument(
        "--batch-size",
        "-b",
        required=False,
        type=int,
        default=8,
        help="The batch size for evaluation.",
    )
    evaluate_parser.add_argument(
        "--k-fold",
        "-k",
        "--parts",
        "-p",
        required=False,
        type=int,
        default=10,
        help="The number of parts to split the dataset into for evaluation.",
    )
    evaluate_parser.add_argument(
        "--single",
        required=False,
        default=False,
        action="store_true",
        help="Whether the model should be evaluated on a single part only.",
    )

    # Parser for the prediction task
    predict_parser = sub_parser.add_parser(str(Tasks.PREDICT))
    predict_parser.add_argument(
        "--model", "-m", required=True, type=Path, help="Path to the model."
    )
    predict_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Path to the snippet."
    )

    return arg_parser


def _run_encode(parsed_args) -> None:
    """
    Runs the encoding of the dataset.
    :param parsed_args: Parsed arguments.
    :return: None
    """
    # Get the parsed arguments
    data_dir = parsed_args.input
    intermediate_dir = parsed_args.intermediate

    # Load the dataset
    raw_data = load_raw_dataset(data_dir)

    # Encode the dataset
    encoded_data = DatasetEncoder().encode_dataset(raw_data)

    # Store the encoded dataset
    if intermediate_dir:
        store_encoded_dataset(encoded_data, intermediate_dir)


def _run_train(parsed_args, model_runner: ModelRunnerInterface) -> None:
    """
    Runs the training of the readability classifier.
    :param parsed_args: Parsed arguments.
    :param model_runner: The model runner.
    :return: None
    """
    # Get the parsed arguments
    data_dir = parsed_args.input
    encoded = parsed_args.encoded
    store_dir = parsed_args.save
    intermediate_dir = parsed_args.intermediate

    # Create the store directory if it does not exist
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    # Load the dataset
    if not encoded:
        raw_data = load_raw_dataset(data_dir)
        encoded_data = DatasetEncoder().encode_dataset(raw_data)

        if intermediate_dir:
            store_encoded_dataset(encoded_data, intermediate_dir)
    else:
        encoded_data = load_encoded_dataset(data_dir)

    # Run the training
    model_runner.run_train(parsed_args, encoded_data)


def _run_predict(parsed_args, model_runner: ModelRunnerInterface) -> tuple[str, float]:
    """
    Runs the prediction of the readability classifier.
    :param parsed_args: Parsed arguments.
    :return: None
    """
    data_input = parsed_args.input

    # Load the snippet
    if not os.path.isfile(data_input):
        raise FileNotFoundError(f"{data_input} does not exist.")

    with open(data_input) as file:
        data_input = file.read()

    logging.info("Loaded Snippet: \n %s", data_input)

    # Encode the snippet
    logging.info("Encoding Snippet...")
    encoded_snippet = DatasetEncoder().encode_text(data_input)

    # Run the prediction
    logging.info("Predicting Readability...")
    return model_runner.run_predict(parsed_args, encoded_snippet)


def _run_evaluate(parsed_args, model_runner: ModelRunnerInterface) -> None:
    """
    Runs the evaluation of the readability classifier.
    :param parsed_args: Parsed arguments.
    :return: None
    """
    data_dir = parsed_args.input
    encoded = parsed_args.encoded
    parts = parsed_args.parts
    single = parsed_args.single

    # Load the dataset
    if not encoded:
        raw_data = load_raw_dataset(data_dir)
        encoded_data = DatasetEncoder().encode_dataset(raw_data)
    else:
        encoded_data = load_encoded_dataset(data_dir)

    # Use only a part of the dataset
    encoded_data = encoded_data.split(split_size=parts)

    if single:
        logging.info(f"Running a single evaluation on {1 / parts}% of the dataset...")
        model_runner.run_evaluate(parsed_args, encoded_data[0])
    else:
        for i, part in enumerate(encoded_data):
            logging.info(f"Running evaluation for part {i + 1}...")
            model_runner.run_evaluate(parsed_args, part)


def main(args: list[str]) -> int:
    """
    Main function of the readability classifier.
    :param args:  List of arguments.
    :return:    Returns 0 if the program was executed successfully.
    """
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    task = Tasks(parsed_args.command)

    # Set up logging and specify logfile name
    logfile = DEFAULT_LOG_FILE
    if hasattr(parsed_args, "save") and parsed_args.save:
        folder_path = Path(parsed_args.save)
        folder_name = Path(parsed_args.save).name
        logfile = folder_path / Path(f"{DEFAULT_LOG_FILE_NAME}-{folder_name}.log")
    _setup_logging(logfile, overwrite=True)

    # Set the seed
    random.seed(SEED)

    # Set up the model runner
    model_runner = KerasModelRunner() if KERAS else TorchModelRunner()

    # Execute the task
    match task:
        case Tasks.ENCODE:
            _run_encode(parsed_args)
        case Tasks.TRAIN:
            _run_train(parsed_args, model_runner)
        case Tasks.PREDICT:
            _run_predict(parsed_args, model_runner)
        case Tasks.EVALUATE:
            _run_evaluate(parsed_args, model_runner)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
