import torch
from torch import nn

# Create a matrix (2D tensor) of input values and set the dtype to float32
X = torch.tensor([[10], [38], [100], [150]], dtype=torch.float32)

# Convert the tensor to int64 dtype
X = X.type(torch.int64)
print(X)
print(X.dtype)

# Will convert the tensor back to float32 dtype
result = X * 0.5
print(result)
print(result.dtype)
