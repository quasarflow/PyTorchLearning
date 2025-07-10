import torch

b = torch.tensor(32)
w1 = torch.tensor(1.8)

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1

# You can get the shape of a tensor using the .shape attribute
print(b.shape)
print(X1.shape)

# You can get the size of a tensor using the .size() method
print(b.size())
print(X1.size())

# You can access elements using indexing and the .item() method
print(y_pred[1].item())
