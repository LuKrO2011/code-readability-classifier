import logging
import os
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any

from readability_classifier.models.model2 import CodeReadabilityClassifier

DEFAULT_LOG_FILE_NAME = "readability-classifier"
DEFAULT_LOG_FILE = f"{DEFAULT_LOG_FILE_NAME}.log"

data_dir = (
    "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Dataset/Dataset/"
)
snippets_dir = os.path.join(data_dir, "Snippets")
csv = os.path.join(data_dir, "scores.csv")


def _setup_logging(log_file: str = DEFAULT_LOG_FILE, overwrite: bool = False) -> None:
    """
    Set up logging.
    """
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

    TRAIN = "TRAIN"
    PREDICT = "PREDICT"
    EVALUATE = "EVALUATE"

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

    train_parser = sub_parser.add_parser(str(Tasks.TRAIN))
    # predict_parser = sub_parser.add_parser(str(Tasks.PREDICT))
    # evaluate_parser = sub_parser.add_parser(str(Tasks.EVALUATE))

    # TODO: Add more params
    train_parser.add_argument("--save", required=False, type=Path)
    # train_parser.add_argument("--delta", required=False, type=int, default=10800)

    return arg_parser


def _run_train(parsed_args) -> None:
    """
    Runs the training of the readability classifier.
    :param parsed_args: Parsed arguments.
    :return: None
    """
    # save = parsed_args.save
    classifier = CodeReadabilityClassifier()
    classifier.prepare_data(csv, snippets_dir)
    classifier.train()


def _run_predict(parsed_args):
    # TODO: Replace with actual arguments
    # save = parsed_args.save

    # TODO: Call actual function
    pass


def _run_evaluate(parsed_args):
    # TODO: Replace with actual arguments
    # save = parsed_args.save

    # TODO: Call actual function
    pass


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
    if parsed_args.save:
        folder_path = Path(parsed_args.save)
        folder_name = Path(parsed_args.save).name
        logfile = folder_path / Path(f"{DEFAULT_LOG_FILE_NAME}-{folder_name}.log")
    _setup_logging(logfile, overwrite=True)

    # Execute the task
    match task:
        case Tasks.TRAIN:
            _run_train(parsed_args)
        case Tasks.PREDICT:
            _run_predict(parsed_args)
        case Tasks.EVALUATE:
            _run_evaluate(parsed_args)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
