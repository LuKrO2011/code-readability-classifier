from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertModel
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

data_dir = 'C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Dataset/Dataset/'
snippets_dir = os.path.join(data_dir, 'Snippets')
csv = os.path.join(data_dir, 'scores.csv')
no_snippets = 200

# Load the CSV file
df = pd.read_csv(csv)

# Extract code snippets and readability scores
code_snippets = []
for i in range(1, no_snippets + 1):
    column_name = f'Snippet{i}'
    code_snippets.append(df[column_name].tolist())

# Drop empty column at the beginning
evaluator_scores = df.drop(columns=['', *code_snippets])

# Calculate the mean score across evaluators for each snippet
aggregated_scores = evaluator_scores.mean(axis=1)

# Tokenize and encode the code snippets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')


def tokenize_and_encode(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embeddings = bert(input_ids)[0]
    return embeddings


embeddings = []
for code_snippet in code_snippets:
    embeddings.append(tokenize_and_encode(code_snippet))

# Split the data into training and test set (X = embeddings, y = aggregated_scores)
X_train, X_test, y_train, y_test = train_test_split(embeddings, aggregated_scores,
                                                    test_size=0.2,
                                                    random_state=42)


# Build and train the model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 768))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1))

        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 49, 128)  # TODO: Adjust
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        # Flatten the feature map
        x = x.view(-1, 64 * 49)  # TODO: Adjust

        # Apply fully connected layers with dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


bert = CNNModel(2)
criterion = nn.MSELoss()
optimizer = optim.Adam(bert.parameters(), lr=0.001)

# TODO: Measure training time and store history weights

# Train the model using X_train, y_train
num_epochs = 10  # TODO: Adjust
batch_size = 32  # TODO: Adjust
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)
for epoch in range(num_epochs):
    bert.train()
    running_loss = 0.0
    # TODO: Add total loss and validation losses

    for i in range(0, len(X_train), batch_size):
        # Send data to cpu/gpu
        # x, y = x.to(device), y.to(device)
        x = torch.Tensor(X_train[i:i + batch_size]).to(device)
        y = torch.Tensor(y_train[i:i + batch_size]).to(device)

        # Forward pass
        outputs = bert(x)

        # Loss calculation
        loss = criterion(outputs, y)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update total loss
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(X_train)}")

# Evaluate the model
bert.eval()
with torch.no_grad():
    # x, y = x.to(device), y.to(device)
    test_inputs = torch.Tensor(X_test).to(device)
    test_labels = torch.Tensor(y_test).to(device)
    predictions = bert(test_inputs)

mse = mean_squared_error(test_labels.cpu().numpy(), predictions.cpu().numpy())
print(f"Mean Squared Error (MSE): {mse}")
