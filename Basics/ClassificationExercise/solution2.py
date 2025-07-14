import torch
from torch import nn
import pandas as pd

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/ClassificationExercise/data/loan_data.csv"
)
df = df[
    [
        "loan_status",
        "person_income",
        "loan_intent",
        "loan_percent_income",
        # "credit_score",
    ]
]
df = pd.get_dummies(df, columns=["loan_intent"])

# Convert the DataFrame to PyTorch tensors
y = torch.tensor(df["loan_status"], dtype=torch.float32).reshape((-1, 1))
X_data = df.drop("loan_status", axis=1).astype("float32").values
X = torch.tensor(X_data, dtype=torch.float32)

# Normalize the features
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X = (X - X_mean) / X_std
print(X.shape)

# Define the model
model = nn.Sequential(
    # Start with 9 or 8 input features and then output 1 value
    nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
)

# Set the loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Set the number of entries and the batch size
num_entries = X.size(0)
batch_size = 32

# Start the training loop
for i in range(0, 100):
    loss_sum = 0
    for start in range(0, num_entries, batch_size):
        end = min(num_entries, start + batch_size)
        X_data = X[start:end]
        y_data = y[start:end]

        optimizer.zero_grad()
        outputs = model(X_data)
        loss = loss_fn(outputs, y_data)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

    if i % 10 == 0:
        print(loss_sum)

# Set the model to evaluation mode and calculate predictions
model.eval()
with torch.no_grad():
    outputs = model(X)
    y_pred = nn.functional.sigmoid(outputs) > 0.5
    y_pred_correct = y_pred.type(torch.float32) == y
    print(y_pred_correct.type(torch.float32).mean())
