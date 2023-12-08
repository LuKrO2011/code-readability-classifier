from enum import Enum
from pathlib import Path
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from readability_classifier.encoders.dataset_utils import ReadabilityDataset
from src.readability_classifier.models.semantic_classifier import SemanticClassifier
from src.readability_classifier.models.structural_classifier import StructuralClassifier
from src.readability_classifier.models.towards_classifier import TowardsClassifier
from src.readability_classifier.models.vi_st_classifier import ViStClassifier
from src.readability_classifier.models.visual_classifier import VisualClassifier
from src.readability_classifier.utils.config import DEFAULT_MODEL_BATCH_SIZE


class ClassifierBuilder:
    _model: nn.Module = None
    _model_path: Path = None
    _train_dataset: ReadabilityDataset = None
    _test_dataset: ReadabilityDataset = None
    _train_loader: DataLoader = None
    _val_loader: DataLoader = None
    _test_loader: DataLoader = None
    _store_dir: Path = None
    _batch_size: int = DEFAULT_MODEL_BATCH_SIZE
    _num_epochs: int = 20
    _learning_rate: float = 0.0015

    def set_model(self, model: nn.Module):
        self._model = model

    def set_datasets(
        self, train_dataset: ReadabilityDataset, test_dataset: ReadabilityDataset
    ):
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

    def set_dataloaders(
        self, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader
    ):
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._val_loader = val_loader

    def set_parameters(
        self, store_dir: Path, batch_size: int, num_epochs: int, learning_rate: float
    ):
        self._store_dir = store_dir
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate

    def set_evaluation_loader(self, test_loader: DataLoader):
        self._test_loader = test_loader

    def set_model_path(self, model_path: Path):
        self._model_path = model_path

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def build(self):
        if self._model == Model.TOWARDS:
            return TowardsClassifier(
                model_path=self._model_path,
                train_dataset=self._train_dataset,
                test_dataset=self._test_dataset,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                test_loader=self._test_loader,
                store_dir=self._store_dir,
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                learning_rate=self._learning_rate,
            )
        if self._model == Model.STRUCTURAL:
            return StructuralClassifier(
                model_path=self._model_path,
                train_dataset=self._train_dataset,
                test_dataset=self._test_dataset,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                test_loader=self._test_loader,
                store_dir=self._store_dir,
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                learning_rate=self._learning_rate,
            )
        if self._model == Model.VISUAL:
            return VisualClassifier(
                model_path=self._model_path,
                train_dataset=self._train_dataset,
                test_dataset=self._test_dataset,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                test_loader=self._test_loader,
                store_dir=self._store_dir,
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                learning_rate=self._learning_rate,
            )
        if self._model == Model.SEMANTIC:
            return SemanticClassifier(
                model_path=self._model_path,
                train_dataset=self._train_dataset,
                test_dataset=self._test_dataset,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                test_loader=self._test_loader,
                store_dir=self._store_dir,
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                learning_rate=self._learning_rate,
            )
        if self._model == Model.VIST:
            return ViStClassifier(
                model_path=self._model_path,
                train_dataset=self._train_dataset,
                test_dataset=self._test_dataset,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                test_loader=self._test_loader,
                store_dir=self._store_dir,
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                learning_rate=self._learning_rate,
            )
        raise ValueError(f"Unknown model {self._model}")


class Model(Enum):
    """
    Enum for the different models.
    """

    TOWARDS = "TOWARDS"
    STRUCTURAL = "STRUCTURAL"
    VISUAL = "VISUAL"
    SEMANTIC = "SEMANTIC"
    VIST = "VIST"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise ModelNotSupportedException(f"{value} is not a supported model.")

    def __str__(self) -> str:
        return self.value


class ModelNotSupportedException(Exception):
    """
    Exception is thrown whenever a model is not supported.
    """
