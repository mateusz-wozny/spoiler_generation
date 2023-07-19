import json
import re
from typing import Dict, List, Optional
import pandas as pd
from nltk.stem import WordNetLemmatizer


class Dataset:
    def __init__(self, df: pd.DataFrame, explode: bool = True) -> None:
        self._df = df
        self.explode = explode
        self.__explode_columns()

    @classmethod
    def from_jsonl(cls, jsonl_path: str, explode: bool = True):
        return cls(pd.read_json(jsonl_path, lines=True), explode)

    @property
    def df(self):
        return self._df

    def __explode_columns(self):
        col = self._df.columns
        if "tags" in col:
            self._df["tags"] = self._df["tags"].explode()
        if "spoiler" in col and self.explode:
            self._df["spoiler"] = self._df["spoiler"].apply(lambda x: "\n".join(x))
        if "postText" in col:
            self._df["postText"] = self._df["postText"].explode()

    def count_posts_by_tags(self) -> Dict[str, int]:
        grouped_df = (
            self._df[["uuid", "tags"]]
            .set_index("tags")
            .groupby(level=0)
            .count()
            .reset_index()
        )
        grouped_df.rename(columns={"uuid": "count"}, inplace=True)
        return grouped_df

    def prepare_post_details(
        self, tag: str, left_idx: int = 0, right_idx: Optional[int] = None
    ) -> pd.DataFrame:
        return self._df.loc[
            self._df["tags"] == tag,
            [
                "postText",
                "spoiler",
                "tags",
                "targetTitle",
                "targetParagraphs",
                "spoilerPositions",
            ],
        ].reset_index(drop=True)[left_idx:right_idx]

    def get_spoiler_text_by_position(self, tag_df: pd.DataFrame) -> List[str]:
        spoiler_text = []
        for idx, (paragraph, position) in enumerate(
            zip(tag_df["targetParagraphs"], tag_df["spoilerPositions"])
        ):
            text = ""
            for points in position:
                if points[0][0] == -1:
                    text += tag_df.iloc[idx, 3][points[0][1] : points[1][1]]
                else:
                    text += paragraph[points[0][0]][points[0][1] : points[1][1]]
            spoiler_text.append(text)
        return spoiler_text

    def prepare_data_for_llm_training(
        self, save_path: Optional[str] = None
    ) -> List[Dict]:
        data = []
        for row in self.df[
            ["targetParagraphs", "postText", "spoiler", "tags"]
        ].itertuples():
            record = {}
            context = row.targetParagraphs
            clickbait = row.postText
            spoilers = row.spoiler

            record["context"] = " ".join(context)
            record["question"] = clickbait
            record["output"] = spoilers
            record["type"] = row.tags

            data.append(record)

        if save_path is not None:
            self.__save_to_json(save_path, data)
        return data

    @staticmethod
    def preprocess_func(x: str) -> str:
        stemmer = WordNetLemmatizer()

        document = re.sub(r"\W", " ", x)
        document = re.sub(r"^b\s+", "", document)

        document = document.lower()
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        return " ".join(document)

    @staticmethod
    def subfinder(mylist, pattern):
        indexes = []
        lengths_list = list(map(len, mylist))

        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i : i + len(pattern)] == pattern:
                indexes.append(
                    [
                        sum(lengths_list[:i]) + i,
                        sum(lengths_list[: i + len(pattern)]) + i + len(pattern) - 1,
                    ]
                )
        return indexes

    def calculate_answer_start(self, split_records: List[str], spoiler: str):
        split_spoiler = spoiler.split()
        return self.subfinder(split_records, split_spoiler)

    def prepare_dataset_for_qa_train(
        self, save_path: Optional[str] = None, create_test_dataset=False
    ) -> List[Dict]:
        data = []
        for row in self.df[
            ["targetParagraphs", "postText", "spoiler", "targetTitle", "tags"]
        ].itertuples():
            record = {}
            context = row.targetParagraphs
            clickbait = row.postText
            spoilers = row.spoiler
            i = str(row.Index + 1)

            record["id"] = "0" * (6 - len(i)) + i
            record["title"] = self.preprocess_func(row.targetTitle)
            record["context"] = self.preprocess_func(" ".join(context))
            record["question"] = self.preprocess_func(clickbait)
            record["type"] = row.tags
            record["answers"] = []
            split_records = record["context"].split()
            if not create_test_dataset:
                for i, spoiler in enumerate(spoilers):
                    spoiler = self.preprocess_func(spoiler)
                    try:
                        result = self.calculate_answer_start(split_records, spoiler)[0]
                    except Exception:
                        continue
                    if i > 5:
                        continue
                    begin_answer, _ = result

                    record["answers"].append(
                        {"text": [spoiler], "answer_start": [begin_answer]}
                    )
                if record["answers"] == []:
                    continue

            data.append(record)

        if save_path is not None:
            self.__save_to_json(save_path, data)
        return data

    def __save_to_json(self, path: str, data: List[Dict]):
        with open(path, "w") as f:
            json.dump(data, f)
