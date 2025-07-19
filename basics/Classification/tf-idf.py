import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/Classification/data/SMSSpamCollection",
    sep="\t",
    names=["type", "message"],
)

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

# This is similar to the CountVectorizer, but it also normalizes the counts
# by the frequency of the words in the document and across all documents.
vectorizer = TfidfVectorizer(max_features=1000)
messages = vectorizer.fit_transform(df["message"])
print(vectorizer.get_feature_names_out()[888])
