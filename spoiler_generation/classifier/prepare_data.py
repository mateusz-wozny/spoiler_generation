import pandas as pd
from datasets import load_dataset

dataset = load_dataset(
    "MateuszW/spoiler_generation",
    data_files={
        "train": "regressor_data/train.csv",
    },
)
df = pd.DataFrame(dataset["train"])
clf_df = pd.DataFrame(columns=["clickbait", "context", "spoiler1", "spoiler2", "class"])
for clickbait, series in df.groupby(by="clickbait"):
    for i in range(series.shape[0]):
        for j in range(series.shape[0]):
            if j == i:
                continue
            if series.iloc[i, 2] != series.iloc[j, 2]:
                if series.iloc[i, 3] == series.iloc[j, 3] == 0:
                    continue
                clf_df = pd.concat(
                    [
                        clf_df,
                        pd.DataFrame(
                            [
                                [
                                    clickbait,
                                    series.iloc[1, 1],
                                    series.iloc[i, 2],
                                    series.iloc[j, 2],
                                    int(series.iloc[i, 3] > series.iloc[j, 3]),
                                ],
                            ],
                            columns=[
                                "clickbait",
                                "context",
                                "spoiler1",
                                "spoiler2",
                                "class",
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

clf_df.to_csv(
    "classifier_data.csv",
    index=False,
)
