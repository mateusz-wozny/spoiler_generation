import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("classifier_data.csv")
print(df.shape)


new_df = pd.DataFrame()
new_df["label"] = df["class"]
new_df["text"] = (
    "For given question:\n"
    + df["clickbait"]
    + "\nchoose what answer is better\n"
    + "\n\n## Answer1:\n"
    + df["spoiler1"]
    + "\n\n## Answer2:\n"
    + df["spoiler2"]
    + "\n\n## Context:\n"
    + df["context"]
)
new_df.dropna(inplace=True)
train, val = train_test_split(new_df, test_size=0.15)
train.to_csv(
    "classifier_train.csv",
    index=False,
)
val.to_csv(
    "classifier_val.csv",
    index=False,
)
