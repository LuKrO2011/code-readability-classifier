from pathlib import Path

import torch
from torch import nn as nn
from transformers import BertConfig, BertModel

from readability_classifier.models.base_model import BaseModel


class KrodBertConfig(BertConfig):
    """
    Configuration class to store the configuration of a `BertEmbedding`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            output_hidden_states=True,
            vocab_size=kwargs.pop("vocab_size", 28996),
            hidden_size=kwargs.pop("hidden_size", 768),
            **kwargs,
        )


class KrodBertEmbedding(BaseModel):
    """
    A Bert embedding layer.
    """

    def __init__(self, config: KrodBertConfig) -> None:
        super().__init__()
        self.model_name = "bert-base-cased"
        self.model = BertModel.from_pretrained(self.model_name, config=config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param inputs: The input tensor.
        :return: The output of the model.
        """
        token_input, segment_input = inputs
        outputs = self.model(token_input, segment_input)
        return outputs.last_hidden_state

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(KrodBertConfig(**params))


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

        self.bert_embedding = KrodBertEmbedding.build_from_config()
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

    def forward(
        self, token_input: torch.Tensor, segment_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param token_input: The token input tensor.
        :param segment_input: The segment input tensor.
        :return: The output of the model.
        """
        texture_embedded = self.bert_embedding([token_input, segment_input])

        # Permute the tensor to fit the convolutional layers
        texture_embedded = texture_embedded.permute(0, 2, 1)

        # Apply convolutional and pooling layers
        x = self.relu(self.conv1(texture_embedded))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output of the conv layers
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
