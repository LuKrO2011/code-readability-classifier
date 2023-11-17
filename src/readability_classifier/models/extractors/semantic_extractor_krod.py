from pathlib import Path

import torch
from torch import nn as nn
from transformers import BertConfig, BertModel

from readability_classifier.models.base_model import BaseModel


class KrodBertConfig(BertConfig):
    """
    Configuration class to store the configuration of a `BertEmbedding`.
    """

    # Use:
    # config = BertConfig.from_pretrained(
    # 'bert-large-uncased', output_hidden_states=True,
    #                                     hidden_dropout_prob=0.2,
    #                                     attention_probs_dropout_prob=0.2)

    def __init__(self, **kwargs):
        super().__init__(
            output_hidden_states=True,
            vocab_size=kwargs.pop("vocab_size", 28996),
            hidden_size=kwargs.pop("hidden_size", 768),
            **kwargs,
        )


class KrodBertEmbedding:
    """
    A Bert embedding layer.
    """

    def __init__(self, config: KrodBertConfig) -> None:
        super().__init__()
        self.model_name = "bert-base-cased"
        self.model = BertModel.from_pretrained(self.model_name)

        # Send the model to the GPU
        self.model.to(torch.device("cuda"))

    def embed(self, input_ids, token_type_ids, attention_mask):
        """
        Forward pass of the model.
        :param inputs: The input tensor.
        :return: The output of the model.
        """
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
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

        self.bert_embedding = KrodBertEmbedding(KrodBertConfig())
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
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param input_ids: The token input tensor.
        :param token_type_ids: The segment input tensor.
        :param attention_mask: The attention mask tensor.
        :return: The output of the model.
        """
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
