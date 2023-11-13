import torch
from torch import device
from torch import nn as nn
from transformers import BertModel

# from src.readability_classifier.models.model import DEFAULT_TOKEN_LENGTH
DEFAULT_TOKEN_LENGTH = 512  # Maximum length of tokens for BERT


class BiLSTM(nn.Module):
    """
    https://www.scaler.com/topics/pytorch/lstm-pytorch/
    """

    def __init__(
        self,
        input_size=DEFAULT_TOKEN_LENGTH,
        hidden_size=128,
        num_layers=3,
        num_classes=10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


class SemanticExtractor(nn.Module):
    """
    A structural feature extractor for code readability classification.
    The model consists of a Bert embedding layer, a convolutional layer, a max pooling
    layer, another convolutional layer and a BiLSTM layer. Relu is used as the
    activation function.
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        # Bert embedding
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=32)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=32)
        self.bilstm = BiLSTM()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        :param input_ids:   Tensor of input_ids for the BERT model.
        :param token_type_ids: Tensor of token_type_ids for the BERT model.
        :param attention_mask: Tensor of attention_mask for the BERT model.
        :return: The output of the model.
        """
        # Bert embedding
        x = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        # Convert the output of the Bert embedding to fitting shape for conv layers
        x = x[0].unsqueeze(1)

        # Apply Layers
        x = nn.ReLU()(self.conv1(x))
        x = self.maxpool(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.bilstm(x)

        return x
