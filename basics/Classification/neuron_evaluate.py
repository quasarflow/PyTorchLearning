import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/Classification/data/SMSSpamCollection",
    sep="\t",
    names=["type", "message"],
)

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

X = torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))

model = nn.Linear(1000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(loss)

model.eval()
with torch.no_grad():
    # Apply the model to the data and get predictions that are greater than 0.25
    y_pred = nn.functional.sigmoid(model(X)) > 0.25

    # Accuracy gives the proportion of correct predictions
    print("Accuracy:", (y_pred == y).type(torch.float32).mean())

    # Sensitivity gives the proportion of actual positives that were correctly predicted
    print("Sensitivity:", (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean())

    # Specificity gives the proportion of actual negatives that were correctly predicted
    print("Specificity:", (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean())

    # Precision gives the proportion of predicted positives that were correct
    print(
        "Precision:", (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean()
    )
