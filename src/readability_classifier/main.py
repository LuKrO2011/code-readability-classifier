import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from readability_classifier.models.model import (
    CodeReadabilityClassifier,
    DatasetEncoder,
    encoded_data_to_dataloaders,
    load_encoded_dataset,
    load_raw_dataset,
)

DEFAULT_LOG_FILE_NAME = "readability-classifier"
DEFAULT_LOG_FILE = f"{DEFAULT_LOG_FILE_NAME}.log"
DEFAULT_MODEL_FILE = "model"


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

    # Parser for the training task
    train_parser = sub_parser.add_parser(str(Tasks.TRAIN))
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
        help="Whether the dataset is already encoded by BERT.",
    )
    train_parser.add_argument(
        "--save",
        "-s",
        required=False,
        type=Path,
        help="Path to the folder where the model should be stored.",
    )
    train_parser.add_argument(
        "--evaluate",
        required=False,
        default=True,
        action="store_false",
        help="Whether the model should be evaluated after training.",
    )
    train_parser.add_argument(
        "--token-length",
        "-l",
        required=False,
        type=int,
        default=512,
        help="The token length of the snippets (cutting/padding applied).",
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
        default=10,
        help="The number of epochs for training.",
    )
    train_parser.add_argument(
        "--learning-rate",
        "-r",
        required=False,
        type=float,
        default=0.0001,
        help="The learning rate for training.",
    )

    # Parser for the evaluation task TODO: Implement
    # evaluate_parser = sub_parser.add_parser(str(Tasks.EVALUATE))

    # Parser for the prediction task
    predict_parser = sub_parser.add_parser(str(Tasks.PREDICT))
    predict_parser.add_argument(
        "--model", "-m", required=True, type=Path, help="Path to the model."
    )
    predict_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Path to the snippet."
    )
    predict_parser.add_argument(
        "--token-length",
        "-l",
        required=False,
        type=int,
        default=512,
        help="The token length of the snippet (cutting/padding applied).",
    )

    return arg_parser


def _run_train(parsed_args) -> None:
    """
    Runs the training of the readability classifier.
    :param parsed_args: Parsed arguments.
    :return: None
    """
    # Get the parsed arguments
    data_dir = parsed_args.input
    encoded = parsed_args.encoded
    store_dir = parsed_args.save
    evaluate = parsed_args.evaluate
    token_length = parsed_args.token_length
    batch_size = parsed_args.batch_size
    num_epochs = parsed_args.epochs
    learning_rate = parsed_args.learning_rate

    if not encoded:
        raw_data = load_raw_dataset(data_dir)
        encoded_data = DatasetEncoder().encode(raw_data)

        # TODO: Add mode for storing encoded dataset
        # store_encoded_dataset(encoded_data, data_dir, token_length)
    else:
        encoded_data = load_encoded_dataset(data_dir)

    train_loader, test_loader = encoded_data_to_dataloaders(encoded_data, batch_size)

    # Train the model
    classifier = CodeReadabilityClassifier(
        train_loader,
        test_loader,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    classifier.train()

    # Evaluate the model
    if evaluate:
        classifier.evaluate()

    # Get the model store name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store_name = f"{DEFAULT_MODEL_FILE}-{token_length}-{batch_size}-{current_time}.pt"
    if store_dir:
        store_name = os.path.join(store_dir, store_name)

    # Store the model
    classifier.store(store_name)


def _run_predict(parsed_args):
    """
    Runs the prediction of the readability classifier.
    :param parsed_args: Parsed arguments.
    :return: None
    """
    # Get the parsed arguments
    model_path = parsed_args.model
    snippet_path = parsed_args.input
    # token_length = parsed_args.token_length

    # Load the model
    classifier = CodeReadabilityClassifier()
    classifier.load(model_path)

    # Predict the readability of the snippet
    with open(snippet_path) as snippet_file:
        snippet = snippet_file.read()
        readability = classifier.predict(snippet)
        logging.info(f"Readability of snippet: {readability}")


def _run_evaluate(parsed_args):
    raise NotImplementedError(
        "Separate evaluation is not implemented. "
        "Please use the --evaluate flag when training."
    )


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
