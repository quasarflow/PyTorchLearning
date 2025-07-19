import sys
import os
import pandas as pd
import torch
from torch import nn

# Pandas: Reading the data
df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/Regression/data/used_cars.csv"
)

# Pandas: Preparing the data
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

# Create a directory for saving the model if it doesn't exist
if not os.path.isdir("/home/jjk339/learn/learn-pytorch/Basics/Regression/model"):
    os.mkdir("/home/jjk339/learn/learn-pytorch/Basics/Regression/model")

# Torch: Creating X and y data (as tensors)
X = torch.column_stack(
    [torch.tensor(age, dtype=torch.float32), torch.tensor(milage, dtype=torch.float32)]
)
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)

# Save the mean and std of X for normalization
torch.save(X_mean, "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/X_mean.pt")
torch.save(X_std, "/home/jjk339/learn/learn-pytorch/Basics/Regression//model/X_std.pt")
X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
y_mean = y.mean()
y_std = y.std()
torch.save(y_mean, "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/y_mean.pt")
torch.save(y_std, "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/y_std.pt")
y = (y - y_mean) / y_std
# sys.exit()

model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0, 2500):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    # if i % 100 == 0:
    #    print(loss)

torch.save(
    model.state_dict(),
    "/home/jjk339/learn/learn-pytorch/Basics/Regression/model/model.pt",
)
