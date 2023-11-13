import torch
from torch import nn as nn
from transformers import BertModel


# TODO: Remove num_classes parameter
class SemanticExtractor(nn.Module):
    """
    A structural feature extractor for code readability classification.
    The model consists of a Bert embedding layer, two convolutional layers,
    two max-pooling layers, two fully connected layers and a dropout layer.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialize the model.
        """
        super().__init__()
        self.num_classes = num_classes

        # Bert embedding
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 768))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1))

        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(8064, 128)  # 8 * 8064 = shape of x
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)

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

        # Apply convolutional and pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        # Flatten the output of the conv layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
