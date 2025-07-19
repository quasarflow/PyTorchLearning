import sys
import pandas as pd
import torch
from torch import nn

X_mean = torch.load(
    "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/X_mean.pt",
    weights_only=True,
)
X_std = torch.load(
    "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/X_std.pt",
    weights_only=True,
)
y_mean = torch.load(
    "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/y_mean.pt",
    weights_only=True,
)
y_std = torch.load(
    "/home/jjk339/learn/learn-pytorch/Basics/Regression//model/y_std.pt",
    weights_only=True,
)

# Create a simple linear regression model and load the state dictionary
model = nn.Linear(2, 1)
model.load_state_dict(
    torch.load(
        "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/model.pt",
        weights_only=True,
    )
)

# Create a test dataset
X_data = torch.tensor([[5, 10000], [2, 10000], [5, 20000]], dtype=torch.float32)

# Normalize the input data and make predictions
model.eval()
with torch.no_grad():
    prediction = model((X_data - X_mean) / X_std)
    print(prediction * y_std + y_mean)
