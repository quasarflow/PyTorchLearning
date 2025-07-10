import torch

# Create two tensors representing the bias and weight
b = torch.tensor(32)
w1 = torch.tensor(1.8)

# Create an array (vector) of input values
X1 = torch.tensor([10, 38, 100, 150])

# Calculate the output of the neuron, tensors have higher priority than scalars
y_pred = 1 * b + X1 * w1
print(y_pred)
