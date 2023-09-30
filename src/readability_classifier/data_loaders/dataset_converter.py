import os
from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset


class CodeLoader(ABC):
    """
    Loads the code snippets from the files.
    """

    @abstractmethod
    def load(self, data_dir: str) -> dict:
        """
        Loads the code snippets from the files to a dictionary. The file names are used
        as keys and the code snippets as values.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The code snippets as a dictionary.
        """
        pass


class ScalabrioCodeLoader(CodeLoader):
    """
    Loads the code snippets of the Scalabrio dataset.
    """

    def load(self, data_dir: str) -> dict:
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


class BWCodeLoader(CodeLoader):
    """
    Loads the code snippets of the BW dataset.
    """

    def load(self, data_dir: str) -> dict:
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
                file_name = file.split(".")[0]
                code_snippets[file_name] = f.read()

        return code_snippets


class DornCodeLoader(CodeLoader):
    """
    Loads the java code snippets of the Dorn dataset.
    """

    def load(self, data_dir: str) -> dict:
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
                file_name = file.split(".")[0]
                code_snippets[file_name] = f.read()

        return code_snippets


class KrodCodeLoader(CodeLoader):
    """
    Loads the java code snippets of the own dataset (krod).
    """

    def __init__(self, name_appendix: str = ""):
        """
        Initializes the code loader.
        """
        super().__init__()
        self.name_appendix = name_appendix

    def load(self, data_dir: str) -> dict:
        """
        Loads the code snippets from the files to a dictionary.
        The path name and file names are used as keys and the code snippets as values.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The code snippets as a dictionary.
        """
        code_snippets = {}

        # Iterate through the files in the directory and subdirectories
        for root, _, files in os.walk(data_dir):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    file_name = os.path.join(root, file) + self.name_appendix
                    code_snippets[file_name] = f.read()

        return code_snippets


class CsvLoader(ABC):
    """
    Loads the ratings data from a CSV file.
    """

    @abstractmethod
    def load(self, csv: str) -> dict:
        """
        Loads the data from the CSV file.
        :param csv: Path to the CSV file containing the scores.
        :return: A dictionary containing the scores.
        """
        pass


class ScalabrioCsvLoader(CsvLoader):
    """
    Loads the ratings data from the Scalabrio CSV file.
    """

    def load(self, csv: str) -> dict:
        """
        Loads the data from the CSV file.
        :param csv: Path to the CSV file containing the scores.
        :return: A dictionary containing the scores.
        """
        data_frame = pd.read_csv(csv)

        # Drop the first column, which contains evaluator names
        data_frame = data_frame.drop(columns=data_frame.columns[0], axis=1)

        # Calculate the mean of the scores for each code snippet
        data_frame = data_frame.mean(axis=0)

        # Turn into dictionary with file names as keys and mean scores as values
        return data_frame.to_dict()


class BWCsvLoader(CsvLoader):
    """
    Loads the ratings data from the BW CSV file.
    """

    def load(self, csv: str) -> dict:
        """
        Loads the data from the CSV file.
        :param csv: Path to the CSV file containing the scores.
        :return: A dictionary containing the scores.
        """
        # Load the data. The first row already contains scores
        data_frame = pd.read_csv(csv, header=None)

        # Drop the first two column, which contains evaluator names
        data_frame = data_frame.drop(columns=data_frame.columns[:2], axis=1)

        # Add a header for all columns (1 - x)
        data_frame.columns = [f"{i}" for i in range(1, len(data_frame.columns) + 1)]

        # Calculate the mean of the scores for each code snippet
        data_frame = data_frame.mean(axis=0)

        # Turn into dictionary with file names as keys and mean scores as values
        return data_frame.to_dict()


class DornCsvLoader(CsvLoader):
    """
    Loads the ratings data from the Dorn CSV file.
    """

    def load(self, csv: str) -> dict:
        """
        Loads the data from the CSV file.
        :param csv: Path to the CSV file containing the scores.
        :return: A dictionary containing the scores.
        """
        # Load the data. The first row already contains scores
        data_frame = pd.read_csv(csv, header=None)

        # Drop the first column, which contains evaluator names
        data_frame = data_frame.drop(columns=data_frame.columns[0], axis=1)

        # Add a header for all columns (1 - x)
        first_file_number = 101
        data_frame.columns = [
            f"{i}"
            for i in range(
                first_file_number, first_file_number + len(data_frame.columns)
            )
        ]

        # Calculate the mean of the scores for each code snippet
        data_frame = data_frame.mean(axis=0)

        # Turn into dictionary with file names as keys and mean scores as values
        return data_frame.to_dict()


class CsvFolderToDataset:
    """
    A data loader for loading data from a CSV file and the corresponding code snippets.
    """

    def __init__(self, csv_loader: CsvLoader, code_loader: CodeLoader):
        """
        Initializes the data loader.
        :param csv_loader: The CSV loader.
        :param code_loader: The code loader.
        """
        self.csv_loader = csv_loader
        self.code_loader = code_loader

    def convert_to_dataset(self, csv: str, data_dir: str) -> Dataset:
        """
        Loads the data and converts it to the HuggingFace format.
        :param csv: Path to the CSV file containing the scores.
        :param data_dir: Path to the directory containing the code snippets.
        :return: The HuggingFace datasets.
        """
        aggregated_scores, code_snippets = self._load_from_storage(csv, data_dir)

        # Combine the scores and the code snippets into a list
        data = []
        for file_name, score in aggregated_scores.items():
            data.append({"code_snippet": code_snippets[file_name], "score": score})

        # Convert to HuggingFace dataset
        return Dataset.from_list(data)

    def _load_from_storage(self, csv: str, data_dir: str) -> tuple[dict, dict]:
        """
        Loads the data from the CSV file and the code snippets from the files.
        :param csv: The path to the CSV file containing the scores.
        :param data_dir: The path to the directory containing the code snippets.
        :return: A tuple containing the mean scores and the code snippets.
        """
        mean_scores = self.csv_loader.load(csv)
        code_snippets = self.code_loader.load(data_dir)

        return mean_scores, code_snippets


class TwoFoldersToDataset:
    """
    A data loader for loading code snippets from two folders and assuming scores.
    """

    def __init__(self, original_loader: CodeLoader, rdh_loader: CodeLoader):
        """
        Initializes the data loader.
        :param original_loader: The code loader.
        """
        self.code_loader = original_loader
        self.rdh_loader = rdh_loader

    def convert_to_dataset(
        self,
        original_data_dir: str,
        rdh_data_dir: str,
        original_score: float = 4.5,
        rdh_score: float = 1.5,
    ) -> Dataset:
        """
        Loads the data and converts it to the HuggingFace format.
        :param original_data_dir: Path to the directory containing the original code
        :param rdh_data_dir: Path to the directory containing the RDH code
        :param original_score: The score for the original code
        :param rdh_score: The score for the RDH code
        :return: The HuggingFace datasets.
        """
        original_code_snippets = self.code_loader.load(original_data_dir)
        rdh_code_snippets = self.rdh_loader.load(rdh_data_dir)

        # Combine the scores and the code snippets into a list
        data = []
        for _, code_snippet in original_code_snippets.items():
            data.append({"code_snippet": code_snippet, "score": original_score})
        for _, code_snippet in rdh_code_snippets.items():
            data.append({"code_snippet": code_snippet, "score": rdh_score})

        # Convert to HuggingFace dataset
        return Dataset.from_list(data)


SCALABRIO_DATA_DIR = (
    "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Dataset/Dataset"
)
BW_DATA_DIR = "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/DatasetBW/"
DORN_DATA_DIR = (
    "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/"
    "DatasetDornJava/dataset"
)

if __name__ == "__main__":
    output_name = "dataset_not_splitted"

    # Get the paths for loading the data
    csv = os.path.join(SCALABRIO_DATA_DIR, "scores.csv")
    snippets_dir = os.path.join(SCALABRIO_DATA_DIR, "Snippets")

    # Load the data
    data_loader = CsvFolderToDataset(
        csv_loader=ScalabrioCsvLoader(), code_loader=ScalabrioCodeLoader()
    )
    dataset = data_loader.convert_to_dataset(csv, snippets_dir)

    # Store the dataset
    dataset.save_to_disk(os.path.join(SCALABRIO_DATA_DIR, output_name))

    # Get the paths for loading the data
    csv = os.path.join(BW_DATA_DIR, "scores.csv")
    snippets_dir = os.path.join(BW_DATA_DIR, "Snippets")

    # Load the data
    data_loader = CsvFolderToDataset(
        csv_loader=BWCsvLoader(), code_loader=BWCodeLoader()
    )
    dataset = data_loader.convert_to_dataset(csv, snippets_dir)

    # Store the dataset
    dataset.save_to_disk(os.path.join(BW_DATA_DIR, output_name))

    # Get the paths for loading the data
    csv = os.path.join(DORN_DATA_DIR, "scores.csv")
    snippets_dir = os.path.join(DORN_DATA_DIR, "Snippets")

    # Load the data
    data_loader = CsvFolderToDataset(
        csv_loader=DornCsvLoader(), code_loader=DornCodeLoader()
    )
    dataset = data_loader.convert_to_dataset(csv, snippets_dir)

    # Store the dataset
    dataset.save_to_disk(os.path.join(DORN_DATA_DIR, output_name))
