import sys
import torch
from torch import nn
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
