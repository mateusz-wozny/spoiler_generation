import os
import pandas as pd
from datasets import load_dataset
import sys

from spoiler_generation.utils.dataset_class import Dataset


def main(args):
    dataset = load_dataset(
        "MateuszW/spoiler_generation",
        data_files={
            "train": "clickbait_spoiling_data/train.jsonl",
            "validation": "clickbait_spoiling_data/validation.jsonl",
        },
    )
    questions = load_dataset(
        "MateuszW/spoiler_generation",
        data_files={
            "train": "generated_questions/train_questions.csv",
            "validation": "generated_questions/validation_questions.csv",
        },
    )
    for part in ["train", "validation"]:
        part_dataset = Dataset(pd.DataFrame(dataset[part]), explode=False)

        part_dataset._df["postText"] = (
            part_dataset._df["postText"] + ". " + questions[part]["generated_questions"]
        )

        part_dataset.prepare_dataset_for_qa_train(
            save_path=os.path.join(args[0], f"{part}.json"), create_test_dataset=False
        )


if __name__ == "__main__":
    main(sys.argv[1:])
