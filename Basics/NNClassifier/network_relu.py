import torch
from torch import nn
import pandas as pd

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/NNClassifier/data/student_exam_data.csv"
)

X = torch.tensor(df[["Study Hours", "Previous Exam Score"]].values, dtype=torch.float32)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1, 1))

# Use the ReLU activation function, which will return 0 for negative inputs
# and the input value for positive inputs.
model = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
print(model)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for i in range(0, 500000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 10000 == 0:
        print(loss)

model.eval()
with torch.no_grad():
    outputs = model(X)
    y_pred = nn.functional.sigmoid(outputs) > 0.5
    y_pred_correct = y_pred.type(torch.float32) == y
    print(y_pred_correct.type(torch.float32).mean())
