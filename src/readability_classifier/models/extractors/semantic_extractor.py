from pathlib import Path

import torch
from torch import nn as nn
from transformers import BertConfig, BertModel

from readability_classifier.models.base_model import BaseModel


class TowardsBertConfig(BertConfig):
    """
    Configuration class to store the configuration of a `BertEmbedding`.
    """

    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=kwargs.pop("vocab_size", 30000),
            type_vocab_size=kwargs.pop("type_vocab_size", 300),
            hidden_size=kwargs.pop("hidden_size", 768),
            num_hidden_layers=kwargs.pop("num_hidden_layers", 12),
            num_attention_heads=kwargs.pop("num_attention_heads", 12),
            intermediate_size=kwargs.pop("intermediate_size", 3072),
            hidden_activation=kwargs.pop("hidden_activation", "gelu"),
            hidden_dropout_rate=kwargs.pop("hidden_dropout_rate", 0.1),
            attention_dropout_rate=kwargs.pop("attention_dropout_rate", 0.1),
            max_position_embeddings=kwargs.pop("max_position_embeddings", 200),
            max_sequence_length=kwargs.pop("max_sequence_length", 100),
            **kwargs,
        )


class BertEmbedding(BaseModel):
    """
    A Bert embedding layer.
    """

    def __init__(self, config: TowardsBertConfig) -> None:
        super().__init__()
        self.model_name = "bert-base-cased"

        # Option 1: Copy weights from pretrained model
        # self.pretrained_model = BertModel.from_pretrained(self.model_name)
        # self.model = BertModel(config=config)
        #
        # # Copy the weights from the pretrained model
        # for name, param in self.pretrained_model.named_parameters():
        #     if name.startswith('bert.embeddings'):
        #         new_param = self.model.state_dict()[name]
        #         if param.data.shape == new_param.shape:
        #             new_param.copy_(param.data)
        #         else:
        #             logging.warning(f"Skipping {name} due to size mismatch")

        # Option 2: Ignore mismatching weights
        self.model = BertModel.from_pretrained(
            "bert-base-cased", config=config, ignore_mismatched_sizes=True
        )

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
        return cls(TowardsBertConfig(**params))


class SemanticExtractorConfig:
    """
    Configuration class to store the configuration of a `SemanticExtractor`.
    """

    def __init__(self, **kwargs: dict[str, ...]):
        # Must be same as hidden_size in BertConfig
        self.input_size = kwargs.get("input_size", 768)


class SemanticExtractor(BaseModel):
    """
    A semantic extractor model. Also known as BertExtractor.
    The model consists of a Bert embedding layer, a convolutional layer, a max pooling
    layer, another convolutional layer and a BiLSTM layer. Relu is used as the
    activation function.
    The input is a tensor of size (1, 512) and the output is a vector of size 10560.
    """

    def __init__(self, config: SemanticExtractorConfig) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        self.bert_embedding = BertEmbedding.build_from_config()

        # In paper: kernel size not specified
        self.conv1 = nn.Conv1d(
            in_channels=config.input_size, out_channels=32, kernel_size=5
        )

        # Same as in paper
        self.relu1 = nn.ReLU()

        # Same as in paper
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)

        # In paper: kernel size not specified
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)

        # In paper: args not specified
        self.bidirectional_lstm = nn.LSTM(32, 32, bidirectional=True, batch_first=True)

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
        texture_embedded = texture_embedded.permute(0, 2, 1)
        x = self.relu1(self.conv1(texture_embedded))
        x = self.maxpool1(x)
        x = self.relu1(self.conv2(x))
        x = x.permute(0, 2, 1)

        x, _ = self.bidirectional_lstm(x)

        # Flatten the tensor after LSTM
        batch_size, seq_length, channels = x.size()
        x = x.contiguous().view(batch_size, -1)  # Flatten the tensor

        return x

    @classmethod
    def _build_from_config(cls, params: dict[str, ...], save: Path) -> "BaseModel":
        """
        Build the model from a config.
        :param params: The config.
        :param save: The path to save the model.
        :return: Returns the model.
        """
        return cls(SemanticExtractorConfig(**params))
