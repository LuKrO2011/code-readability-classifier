from dataclasses import dataclass

import torch

DEFAULT_MODEL_BATCH_SIZE = 8  # Small - avoid CUDA out of memory errors on local machine


@dataclass(frozen=True)
class ModelInput:
    """
    Abstract data class for the input of a model.
    """


@dataclass(frozen=True)
class TowardsInput(ModelInput):
    """
    Data class for the input of the TowardsModel.
    """

    character_matrix: torch.Tensor
    token_input: torch.Tensor
    segment_input: torch.Tensor
    image: torch.Tensor


@dataclass(frozen=True)
class StructuralInput(ModelInput):
    """
    Data class for the input of the StructuralModel.
    """

    character_matrix: torch.Tensor
