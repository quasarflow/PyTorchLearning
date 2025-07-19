import torch
import pandas as pd

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/ClassificationExercise/data/loan_data.csv"
)
df = df[
    [
        "loan_status",
        "person_income",
        "loan_intent",
        "loan_percent_income",
        "credit_score",
    ]
]
# Perform the one-hot encoding for the categorical variable
df = pd.get_dummies(df, columns=["loan_intent"])
print(df.columns)

# Set the target variable
y = torch.tensor(df["loan_status"], dtype=torch.float32).reshape((-1, 1))
print(df.drop("loan_status", axis=1))

# Drop the target variable from the features and convert to float32
X_data = df.drop("loan_status", axis=1).astype("float32").values
print(X_data.dtype)

# Turn the features into a tensor
X = torch.tensor(X_data, dtype=torch.float32)
print(X)
