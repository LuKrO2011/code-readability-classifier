import os
from pathlib import Path

import torch
from torch import nn as nn
from transformers import BertConfig, BertModel

from readability_classifier.encoders.bert_encoder import DEFAULT_OWN_SEGMENT_IDS
from src.readability_classifier.toch.base_model import BaseModel
from src.readability_classifier.utils.config import SemanticInput
from src.readability_classifier.utils.utils import load_yaml_file

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


class KrodBertEmbedding(BaseModel):
    """
    A Bert embedding layer.
    """

    def __init__(self, config: KrodBertConfig) -> None:
        super().__init__()
        self.model_name = "bert-base-cased"
        self.model = BertModel.from_pretrained(self.model_name, config=config)
        if DEFAULT_OWN_SEGMENT_IDS:
            self.model.resize_token_embeddings(config.vocab_size + 1)

        # Send the model to the GPU
        self.model.to(self.device)

    def forward(
        self,
        x: SemanticInput,
    ) -> torch.Tensor:
        """
        Embed the input using Bert.
        :param x: The input.
        :return:
        """
        input_ids = x.input_ids
        token_type_ids = x.token_type_ids
        attention_mask = x.attention_mask
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

        self.bert_embedding = KrodBertEmbedding.build_from_config()
        self.relu = nn.ReLU()

        # Convolutional layers
        # self.conv1 = nn.Conv1d(
        #     in_channels=768, out_channels=32, kernel_size=5)
        # self.pool1 = nn.MaxPool1d(kernel_size=5)
        # self.conv2 = nn.Conv1d(
        #     in_channels=32, out_channels=32, kernel_size=5)

        # Bidirectional LSTM
        # self.lstm = nn.LSTM(input_size=32, hidden_size=32, bidirectional=True)

        self.conv1 = nn.Conv1d(
            in_channels=config.input_size, out_channels=32, kernel_size=2
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.flatten = nn.Flatten()

    def forward(self, x: SemanticInput) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: The input of the model.
        :return: The output of the model.
        """
        # Shape of bert output: (batch_size, sequence_length, hidden_size)
        with torch.no_grad():
            texture_embedded = self.bert_embedding(x)

        # Permute the tensor to fit the convolutional layer
        # -> (batch_size, hidden_size, sequence_length)
        x = texture_embedded.permute(0, 2, 1)

        # # Apply convolutional and pooling layers
        # x = self.relu(self.conv1(x))
        # x = self.pool1(x)
        # x = self.relu(self.conv2(x))
        #
        # # Permute the tensor to fit the LSTM layer
        # # -> (batch_size, sequence_length, hidden_size)
        # x = x.permute(0, 2, 1)
        #
        # # Apply LSTM
        # x, _ = self.lstm(x)
        #
        # # Flatten the tensor after LSTM
        # # -> (batch_size, sequence_length * hidden_size)
        # x = self.flatten(x)

        # Apply convolutional and pooling layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output of the conv layers
        return self.flatten(x)

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(KrodSemanticExtractorConfig(**params))
