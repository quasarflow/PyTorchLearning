import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/ClassificationExercise/data/loan_data.csv"
)
# Select specific columns from the DataFrame
df = df[
    [
        "loan_status",
        "person_income",
        "loan_intent",
        "loan_percent_income",
        "credit_score",
    ]
]
