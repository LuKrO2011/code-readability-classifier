import logging
import os
import random
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any
from readability_classifier.model_runner_interface import ModelRunnerInterface
from readability_classifier.encoders.dataset_encoder import DatasetEncoder
from src.readability_classifier.keas.model_runner import KerasModelRunner

DEFAULT_LOG_FILE_NAME = "readability-classifier"
DEFAULT_LOG_FILE = f"{DEFAULT_LOG_FILE_NAME}.log"
DEFAULT_MODEL_FILE = "model"
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SAVE_DIR = os.path.join(CURR_DIR, "../../models")
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

    # Parser for the prediction task
    predict_parser = sub_parser.add_parser(str(Tasks.PREDICT))
    predict_parser.add_argument(
        "--model", "-m", required=True, type=Path, help="Path to the model."
    )
    predict_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Path to the snippet."
    )

    return arg_parser


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
    model_runner = KerasModelRunner()

    # Execute the task
    match task:
        case Tasks.PREDICT:
            _run_predict(parsed_args, model_runner)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
