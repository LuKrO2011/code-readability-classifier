import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from readability_classifier.models.fc_model import FullyConnectedModel
from readability_classifier.utils.config import BaseModelConfig, ModelInput
from readability_classifier.utils.utils import load_yaml_file

CURR_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
DEFAULT_SAVE_PATH = CURR_DIR / Path("../../../models/")


# TODO: Save not needed here?
class BaseModel(nn.Module, ABC):
    """
    Abstract base class for a model.
    """

    CURR_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
    CONFIGS_PATH = CURR_DIR / Path("../../res/models/")
    SAVE_PATH = CURR_DIR / Path("../../../models/")
    TRAIN_STATS_FILE = "train_stats.json"

    def __init__(self) -> None:
        """
        Initializes the model.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, x: ModelInput) -> torch.Tensor:
        """
        Forward pass.
        :param x: The input.
        :return: The output.
        """
        pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path) -> "BaseModel":
        """
        Load the model from the given path.
        :param checkpoint_path: The path to the checkpoint.
        :return: Returns the loaded model.
        """
        model = cls.build_from_config()
        model.load_state_dict(torch.load(checkpoint_path))
        return model

    @classmethod
    def build_from_config(cls, save: Path = DEFAULT_SAVE_PATH) -> "BaseModel":
        """
        Builds the model from the models file params.
        It also sets the checkpoint path.
        :param save: The path to store the checkpoints.
        :return: Returns the loaded model.
        """
        return cls._build_from_config(cls._get_model_params(), save)

    @classmethod
    def _get_model_params(cls) -> dict[str, ...]:
        """
        Get the model hyperparams.
        :return: The hyperparams.
        """
        return load_yaml_file(cls.CONFIGS_PATH / f"{cls.__name__.lower()}.yaml")

    @classmethod
    @abstractmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Sets the model hyperparams from the models file.
        Also sets the checkpoint path.
        :param params: The hyperparams.
        :param save: The path to store the checkpoints.
        :return: Returns the loaded model.
        """
        pass

    def _build_classification_layers(self, config: BaseModelConfig) -> None:
        """
        Defines the own classification layers of the model.
        :param config: The config for the model.
        """
        self.fc_model = FullyConnectedModel(config)

    def _forward_classification_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the classification layers.
        :param x: The input.
        :return: The output.
        """
        return self.fc_model(x)

    def _update_input_length(self, input_length: int) -> None:
        """
        Update the input length of the forward classification layers, if needed.
        :param input_length: The new input length.
        """
        self.fc_model.update_input_length(input_length, self.device)
