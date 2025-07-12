import torch
from torch import nn
import pandas as pd

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/NNClassifier/data/student_exam_data.csv"
)

X = torch.tensor(df[["Study Hours", "Previous Exam Score"]].values, dtype=torch.float32)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1, 1))

# Define a hidden layer that will take 2 inputs and output 10 features
# Define an output layer that will take 10 inputs and output 1 feature
hidden_model = nn.Linear(2, 10)
output_model = nn.Linear(10, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
# The parameters of the model are the weights and biases of the hidden and output layers
parameters = list(hidden_model.parameters()) + list(output_model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.005)

for i in range(0, 500000):
    optimizer.zero_grad()
    outputs = hidden_model(X)
    # Apply the sigmoid activation function to the outputs of the hidden layer
    outputs = nn.functional.sigmoid(outputs)
    outputs = output_model(outputs)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 10000 == 0:
        print(loss)

# model.eval()
# with torch.no_grad():
#    y_pred = nn.functional.sigmoid(model(X)) > 0.5
#    y_pred_correct = y_pred.type(torch.float32) == y
#    print(y_pred_correct.type(torch.float32).mean())
