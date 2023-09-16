import os

import pandas as pd
from datasets import Dataset, DatasetDict


class CsvFolderToDataset:
    """
    A data loader for loading data from a CSV file and the corresponding code snippets.
    """

    def __init__(self):
        """
        Initializes the data loader.
        """

    def convert_to_dataset(self, csv: str, data_dir: str) -> DatasetDict:
        """
        Loads the data and converts it to the HuggingFace format.
        :param csv: Path to the CSV file containing the scores.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The HuggingFace datasets.
        """
        aggregated_scores, code_snippets = self._load_from_storage(csv, data_dir)

        # Combine the scores and the code snippets into a single dictionary
        data = {}
        for file_name, score in aggregated_scores.items():
            data[file_name] = {"code": code_snippets[file_name], "score": score}

        # Convert to HuggingFace format
        dataset = Dataset.from_dict(data)

        # Split into train and test
        dataset = dataset.train_test_split(test_size=0.2)

        return dataset

    def _load_from_storage(self, csv: str, data_dir: str) -> tuple[dict, dict]:
        """
        Loads the data from the CSV file and the code snippets from the files.
        :param csv: The path to the CSV file containing the scores.
        :param data_dir: The path to the directory containing the code snippets.
        :return: A tuple containing the mean scores and the code snippets.
        """
        mean_scores = self._load_mean_scores(csv)
        code_snippets = self._load_code_snippets(data_dir)

        return mean_scores, code_snippets

    def _load_code_snippets(self, data_dir: str) -> dict:
        """
        Loads the code snippets from the files to a dictionary. The file names are used
        as keys and the code snippets as values.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The code snippets as a dictionary.
        """
        code_snippets = {}

        # Iterate through the files in the directory
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file)) as f:
                # Replace "1.jsnp" with "Snippet1" etc. to match file names in the CSV
                file_name = file.split(".")[0]
                file_name = f"Snippet{file_name}"
                code_snippets[file_name] = f.read()

        return code_snippets

    def _load_mean_scores(self, csv: str) -> dict:
        """
        Loads the mean scores from the CSV file.
        :param csv: Path to the CSV file containing the scores.
        :return: A pandas Series containing the mean scores.
        """
        data_frame = pd.read_csv(csv)

        # Drop the first column, which contains evaluator names
        data_frame = data_frame.drop(columns=data_frame.columns[0], axis=1)

        # Calculate the mean of the scores for each code snippet
        data_frame = data_frame.mean(axis=0)

        # Turn into dictionary with file names as keys and mean scores as values
        return data_frame.to_dict()


DATA_DIR = (
    "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Dataset/Dataset/"
)

if __name__ == "__main__":
    # Get the paths for loading the data
    csv = os.path.join(DATA_DIR, "scores.csv")
    snippets_dir = os.path.join(DATA_DIR, "Snippets")

    # Load the data
    data_loader = CsvFolderToDataset()
    dataset = data_loader.convert_to_dataset(csv, snippets_dir)

    # Store the dataset
    dataset.save_to_disk(os.path.join(DATA_DIR, "dataset"))
