import torch

# Create a matrix (2D tensor) of input values
X = torch.tensor([[10], [38], [100], [150]])

# You can get the size of the matrix, in this case the columns
print(X.size(1))

# Print all rows of the first column
print(X[:, 0])
