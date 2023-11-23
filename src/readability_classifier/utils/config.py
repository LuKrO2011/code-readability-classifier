from dataclasses import dataclass

import torch

DEFAULT_MODEL_BATCH_SIZE = 8  # Small - avoid CUDA out of memory errors on local machine


@dataclass(frozen=False)
class BaseModelConfig:
    """
    Abstract data class for the configuration of a model.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the configuration.
        :param kwargs: The configuration parameters.
        """
        self.input_length = kwargs.get("input_length")
        self.output_length = kwargs.get("output_length")
        self.dropout = kwargs.get("dropout")


@dataclass(frozen=True)
class ModelInput:
    """
    Abstract data class for the input of a model.
    """


@dataclass(frozen=True)
class StructuralInput(ModelInput):
    """
    Data class for the input of the StructuralModel.
    """

    character_matrix: torch.Tensor


@dataclass(frozen=True)
class VisualInput(ModelInput):
    """
    Data class for the input of the VisualModel.
    """

    image: torch.Tensor


@dataclass(frozen=True)
class SemanticInput(ModelInput):
    """
    Data class for the input of the SemanticModel.
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    segment_ids: torch.Tensor


@dataclass(frozen=True)
class TowardsInput(ModelInput):
    """
    Data class for the input of the TowardsModel.
    """

    character_matrix: torch.Tensor
    bert: SemanticInput
    image: torch.Tensor


@dataclass(frozen=True)
class ViStModelInput(ModelInput):
    """
    Data class for the input of the ViStModel.
    """

    image: torch.Tensor
    matrix: torch.Tensor
