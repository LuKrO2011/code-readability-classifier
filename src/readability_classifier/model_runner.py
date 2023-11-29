import logging
from abc import ABC, abstractmethod

from readability_classifier.model_buider import ClassifierBuilder
from readability_classifier.models.encoders.dataset_utils import (
    ReadabilityDataset,
    dataset_to_dataloader,
    load_encoded_dataset,
    split_train_test,
    split_train_val,
)
from readability_classifier.models.towards_classifier import TowardsClassifier


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
    def run_predict(self, parsed_args):
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        pass

    @abstractmethod
    def run_evaluate(self, parsed_args):
        """
        Runs the evaluation of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        pass


class TorchModelRunner(ModelRunnerInterface):
    """
    A torch model runner. Runs the training, prediction and evaluation of a
    PyTorch readability classifier.
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
        # Get the parsed arguments
        model = parsed_args.model
        store_dir = parsed_args.save
        evaluate = parsed_args.evaluate
        batch_size = parsed_args.batch_size
        num_epochs = parsed_args.epochs
        learning_rate = parsed_args.learning_rate

        # Split the dataset
        train_test = split_train_test(encoded_data)
        train_dataset, test_dataset = train_test.train_set, train_test.test_set
        train_val = split_train_val(train_dataset)
        train_dataset, test_dataset = train_val.train_set, train_val.val_set
        train_loader = dataset_to_dataloader(train_dataset, batch_size)
        val_loader = dataset_to_dataloader(test_dataset, batch_size)
        test_loader = dataset_to_dataloader(train_test.test_set, batch_size)

        # Build the model
        builder = ClassifierBuilder()
        builder.set_model(model)
        builder.set_dataloaders(train_loader, test_loader, val_loader)
        builder.set_parameters(store_dir, batch_size, num_epochs, learning_rate)
        classifier = builder.build()

        # Train the model
        classifier.fit()

        # Evaluate the model
        if evaluate:
            classifier.evaluate()

    def _run_with_cross_validation(self, parsed_args, encoded_data: ReadabilityDataset):
        """
        Runs the training of the readability classifier with cross-validation.
        :param parsed_args: Parsed arguments.
        :param encoded_data: The encoded dataset.
        :return: None
        """
        # Get the parsed arguments
        model = parsed_args.model
        store_dir = parsed_args.save
        batch_size = parsed_args.batch_size
        num_epochs = parsed_args.epochs
        learning_rate = parsed_args.learning_rate

        # Build the model
        train_test = split_train_test(encoded_data)
        train_dataset, test_dataset = train_test.train_set, train_test.test_set

        # Build the model
        builder = ClassifierBuilder()
        builder.set_model(model)
        builder.set_datasets(train_dataset, test_dataset)
        builder.set_parameters(store_dir, batch_size, num_epochs, learning_rate)
        classifier = builder.build()

        # Train the model
        classifier.k_fold_cv()

    def run_predict(self, parsed_args):
        """
        Runs the prediction of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        # Get the parsed arguments
        model_path = parsed_args.model
        snippet_path = parsed_args.input

        # Load the model
        classifier = TowardsClassifier()
        classifier.load(model_path)

        # Predict the readability of the snippet
        with open(snippet_path) as snippet_file:
            snippet = snippet_file.read()
            readability = classifier.predict(snippet)
            logging.info(f"Readability of snippet: {readability}")

    def run_evaluate(self, parsed_args):
        """
        Runs the evaluation of the readability classifier.
        :param parsed_args: Parsed arguments.
        :return: None
        """
        # Get the parsed arguments
        model_path = parsed_args.load
        data_dir = parsed_args.input
        model = parsed_args.model
        batch_size = parsed_args.batch_size

        # Load the dataset
        encoded_data = load_encoded_dataset(data_dir)

        # TODO: Split once and store the split
        logging.warning("The test set used is not unseen data!")
        test_dataset = split_train_test(encoded_data).test_set
        test_loader = dataset_to_dataloader(test_dataset, batch_size)

        # Load the model
        builder = ClassifierBuilder()
        builder.set_model(model)
        builder.set_evaluation_loader(test_loader)
        builder.set_model_path(model_path)
        builder.set_batch_size(batch_size)
        classifier = builder.build()

        # Evaluate the model
        classifier.evaluate()
