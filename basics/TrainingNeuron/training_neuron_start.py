import torch
from torch import nn

X1 = torch.tensor([[10.0]])  # Input: Temperature in °C
y1 = torch.tensor([[50.0]])  # Actual value: Temperature °F

X2 = torch.tensor([[37.78]])  # Input: Temperature in °C
y2 = torch.tensor([[100.0]])  # Actual value: Temperature °F

model = nn.Linear(1, 1)

"""
The way that a neuron learns is by adjusting its weights and biases.
We need a metric that can quantify how far off the neuron's output is from the actual value.

We use whats called a loss function, which will take the predicted output and actual value
and then will return a single number that represents how far off the prediction is.
A standard loss function for regression tasks is Mean Squared Error (MSE).

The MSE loss function calculates the average of the squares of the errors, which is the difference
between the predicted and actual values. L = 1/n * Σ(actual - predicted)²
"""
loss_fn = torch.nn.MSELoss()

"""
The optimizer is responsible for updating the weights and biases of the model based on the gradients calculated during backpropagation.
It uses a learning rate to determine how much to adjust the weights and biases, which is usually between 0.0001 and 0.1.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for i in range(0, 50000):
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(model.weight)
        print(model.bias)
