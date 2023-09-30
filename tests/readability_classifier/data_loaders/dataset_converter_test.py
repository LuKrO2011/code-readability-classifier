import os
import unittest
from tempfile import TemporaryDirectory

from src.readability_classifier.data_loaders.dataset_converter import (
    BWCodeLoader,
    BWCsvLoader,
    CsvFolderToDataset,
    DornCodeLoader,
    DornCsvLoader,
    KrodCodeLoader,
    ScalabrioCodeLoader,
    ScalabrioCsvLoader,
    TwoFoldersToDataset,
)


class TestDataConversion(unittest.TestCase):
    output_dir = None  # Set to "output" to generate output
    test_data_dir = "res/raw_data"

    def setUp(self):
        # Create temporary directories for testing if output directory is None
        if self.output_dir is None:
            self.temp_dir = TemporaryDirectory()
            self.output_dir = self.temp_dir.name
        else:
            self.temp_dir = None

    def tearDown(self):
        # Clean up temporary directories
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

    def test_ScalabrioDataConversion(self):
        # Test loading and saving Scalabrio dataset
        data_dir = os.path.join(self.test_data_dir, "scalabrio")
        csv = os.path.join(data_dir, "scores.csv")
        snippets_dir = os.path.join(data_dir, "Snippets")

        # Load the data
        data_loader = CsvFolderToDataset(
            csv_loader=ScalabrioCsvLoader(), code_loader=ScalabrioCodeLoader()
        )
        dataset = data_loader.convert_to_dataset(csv, snippets_dir)

        # Store the dataset
        dataset.save_to_disk(self.output_dir)

        # Check if the dataset was saved successfully
        assert os.path.exists(self.output_dir)

    def test_BWDataConversion(self):
        # Test loading and saving BW dataset
        data_dir = os.path.join(self.test_data_dir, "bw")
        csv = os.path.join(data_dir, "scores.csv")
        snippets_dir = os.path.join(data_dir, "Snippets")

        # Load the data
        data_loader = CsvFolderToDataset(
            csv_loader=BWCsvLoader(), code_loader=BWCodeLoader()
        )
        dataset = data_loader.convert_to_dataset(csv, snippets_dir)

        # Store the dataset
        dataset.save_to_disk(self.output_dir)

        # Check if the dataset was saved successfully
        assert os.path.exists(self.output_dir)

    def test_DornDataConversion(self):
        # Test loading and saving Dorn dataset
        data_dir = os.path.join(self.test_data_dir, "dorn")
        csv = os.path.join(data_dir, "scores.csv")
        snippets_dir = os.path.join(data_dir, "Snippets")

        # Load the data
        data_loader = CsvFolderToDataset(
            csv_loader=DornCsvLoader(), code_loader=DornCodeLoader()
        )
        dataset = data_loader.convert_to_dataset(csv, snippets_dir)

        # Store the dataset
        dataset.save_to_disk(self.output_dir)

        # Check if the dataset was saved successfully
        assert os.path.exists(self.output_dir)

    def test_KrodingerDataConversion(self):
        # Test loading and saving Krodinger dataset
        data_dir = os.path.join(self.test_data_dir, "krod")
        original = os.path.join(data_dir, "original")
        rdh = os.path.join(data_dir, "rdh")

        # Load the data
        data_loader = TwoFoldersToDataset(
            original_loader=KrodCodeLoader(),
            rdh_loader=KrodCodeLoader(name_appendix="rdh"),
        )
        dataset = data_loader.convert_to_dataset(original, rdh)

        # Store the dataset
        dataset.save_to_disk(self.output_dir)

        # Check if the dataset was saved successfully
        assert os.path.exists(self.output_dir)
