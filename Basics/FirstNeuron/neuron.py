import torch
from torch import nn

# Create an input matrix (2D tensor) of input values
X = torch.tensor([[10.0], [38.0], [100.0], [150.0]])

# Create a simple linear model with one input and one output
model = nn.Linear(1, 1)

# Initialize the model parameters
model.bias = nn.Parameter(torch.tensor([32.0]))

# The weight will be a matrix since it can have multiple features, but here we have only one feature
model.weight = nn.Parameter(torch.tensor([[1.8]]))

print(model)
print(model.bias)
print(model.weight)

# Forward pass through the model
y_pred = model(X)
print(y_pred)
