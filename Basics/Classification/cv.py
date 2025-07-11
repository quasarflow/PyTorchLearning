import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv(
    "/home/jjk339/learn/learn-pytorch/Basics/Classification/data/SMSSpamCollection",
    sep="\t",
    names=["type", "message"],
)

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

# The CountVectorizer converts text documents to a matrix of token counts.
# Set the maximum number of features to 1000.
cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])
print(cv.get_feature_names_out()[888])

# ----
# cv = CountVectorizer(max_features=6)
# documents = [
#     "Hello world. Today is amazing. Hello hello",
#     "Hello mars, today is perfect"
# ]
# cv.fit(documents)
# print(cv.get_feature_names_out())
# out = cv.transform(documents)
# print(out.todense())
