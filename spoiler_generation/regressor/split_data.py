import pandas as pd
from sklearn.model_selection import train_test_split

from spoiler_generation.spoiler_generation.utils.dataset_class import Dataset

df = pd.read_csv("spoiler_generation/regressor/best_models_val_data/regressor_data.csv")
print(df.shape)


new_df = pd.DataFrame()
new_df["bleu"] = df["bleu"].astype(float)
df.dropna(inplace=True)
df["spoiler"] = df["spoiler"].apply(Dataset.preprocess_func)
new_df["text"] = "For given question:\n" + df["clickbait"] + "\nanswer:\n" + df["spoiler"] + "\ncontext:\n" + df["context"]
new_df.dropna(inplace=True)
train, val = train_test_split(new_df, test_size=0.15)
train.to_csv(
    "spoiler_generation/regressor/best_models_val_data/train.csv",
    index=False,
)
val.to_csv(
    "spoiler_generation/regressor/best_models_val_data/val.csv",
    index=False,
)
