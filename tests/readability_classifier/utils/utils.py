import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

CURR_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
RES_DIR = CURR_DIR / "../../res"

RAW_DATASETS_DIR = RES_DIR / "raw_datasets"
RAW_SCALABRIO_DIR = RAW_DATASETS_DIR / "scalabrio"
RAW_BW_DIR = RAW_DATASETS_DIR / "bw"
RAW_DORN_DIR = RAW_DATASETS_DIR / "dorn"
RAW_COMBINED_DIR = RAW_DATASETS_DIR / "combined"

ENCODED_DATASETS_DIR = RES_DIR / "encoded_datasets"
ENCODED_SCALABRIO_DIR = ENCODED_DATASETS_DIR / "scalabrio"
ENCODED_BW_DIR = ENCODED_DATASETS_DIR / "bw"
ENCODED_DORN_DIR = ENCODED_DATASETS_DIR / "dorn"
ENCODED_COMBINED_DIR = ENCODED_DATASETS_DIR / "combined"

MI_DIR = RES_DIR / "mi"
MI_RAW_DIR = MI_DIR / "raw"
MI_STRUCTURAL_DIR = MI_DIR / "structural"

MODELS_DIR = RES_DIR / "models"
TOWARDS_MODEL = MODELS_DIR / "towards.keras"

RAW_DATA_DIR = RES_DIR / "raw_data"
BW_SNIPPET_1 = RAW_DATA_DIR / "bw/Snippets/1.jsnp"

HISTORY_DIR = RES_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "history.pkl"
STATS_FILE = HISTORY_DIR / "stats.json"

CODE_SNIPPETS_DIR = RES_DIR / "code_snippets"
TOWARDS_CODE_SNIPPET = CODE_SNIPPETS_DIR / "towards.java"
DIR_WITH_ONE_SNIPPET = CODE_SNIPPETS_DIR / "AreaShop/AreaShopInterface.java"
DIR_WITH_FOUR_SNIPPETS = CODE_SNIPPETS_DIR / "AreaShop/AddCommand.java"


class DirTest(unittest.TestCase):
    output_dir_name = None  # Set to "output" to generate output

    def setUp(self):
        # Create temporary directories for testing if output directory is None
        if self.output_dir_name is None:
            self._temp_dir = TemporaryDirectory()
            self.output_dir = self._temp_dir.name
        else:
            Path(self.output_dir_name).mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
            self.output_dir = self.output_dir_name

    def tearDown(self):
        # Clean up temporary directories
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
