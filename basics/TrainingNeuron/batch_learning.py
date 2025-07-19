import torch
from torch import nn

# Input: Temperature in °C
X = torch.tensor([[10], [37.78]], dtype=torch.float32)

# Actual value: Temperature °F
y = torch.tensor([[50], [100.0]], dtype=torch.float32)

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for i in range(0, 150000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(model.bias)
        print(model.weight)

print("----")

# Create a test input
measurements = torch.tensor([[37.5]], dtype=torch.float32)

# Put the model in evaluation mode, which disables dropout and batch normalization
model.eval()
# Perform inference without tracking gradients
with torch.no_grad():
    prediction = model(measurements)
    print(prediction)  # Should be close to 99.5
