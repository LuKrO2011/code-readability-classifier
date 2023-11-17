import os
from pathlib import Path

import torch
from torch import nn as nn
from transformers import BertConfig, BertModel

from readability_classifier.models.base_model import BaseModel
from readability_classifier.utils.config import SemanticInput
from readability_classifier.utils.utils import load_yaml_file

CURR_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
DEFAULT_SAVE_PATH = CURR_DIR / Path("../../../models/")
CONFIGS_PATH = CURR_DIR / Path("../../res/models/")


class KrodBertConfig(BertConfig):
    """
    Configuration class to store the configuration of a `BertEmbedding`.
    """

    def __init__(self, **kwargs):
        if "vocab_size" not in kwargs:
            kwargs["vocab_size"] = 28996
        super().__init__(**kwargs)

    @classmethod
    def build_config(cls) -> "KrodBertConfig":
        """
        Build the model from a config.
        :return: Returns the model.
        """
        return cls(**cls._get_config())

    @classmethod
    def _get_config(cls) -> dict[str, ...]:
        """
        Get the config hyperparams.
        :return: The hyperparams.
        """
        return load_yaml_file(CONFIGS_PATH / f"{cls.__name__.lower()}.yaml")


class KrodBertEmbedding:
    """
    A Bert embedding layer.
    """

    def __init__(self, config: KrodBertConfig) -> None:
        super().__init__()
        self.model_name = "bert-base-cased"
        self.model = BertModel.from_pretrained(self.model_name, config=config)

        # Send the model to the GPU
        self.model.to(torch.device("cuda"))

    def embed(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed the input using Bert.
        :param input_ids: The token input tensor.
        :param token_type_ids: The segment input tensor.
        :param attention_mask: The attention mask tensor.
        :return:
        """
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


class KrodSemanticExtractorConfig:
    """
    Configuration class to store the configuration of a `SemanticExtractor`.
    """

    def __init__(self, **kwargs: dict[str, ...]):
        # Must be same as hidden_size in BertConfig
        self.input_size = kwargs.get("input_size", 768)


class KrodSemanticExtractor(BaseModel):
    """
    A semantic extractor model. Also known as BertExtractor.
    """

    def __init__(self, config: KrodSemanticExtractorConfig) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        self.bert_embedding = KrodBertEmbedding(KrodBertConfig.build_config())
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(
            in_channels=config.input_size, out_channels=32, kernel_size=5
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

    def forward(self, x: SemanticInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model.
        :return: The output of the model.
        """
        input_ids = x.input_ids
        token_type_ids = x.token_type_ids
        attention_mask = x.attention_mask

        # Shape of bert output: (batch_size, sequence_length, hidden_size)
        texture_embedded = self.bert_embedding.embed(
            input_ids, token_type_ids, attention_mask
        )

        # Shape of texture_embedded: (batch_size, hidden_size, sequence_length)
        texture_embedded = texture_embedded.transpose(1, 2)

        # Apply convolutional and pooling layers
        x = self.relu(self.conv1(texture_embedded))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten
        x = self.flatten(x)

        return x

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(KrodSemanticExtractorConfig(**params))