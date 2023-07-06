import pandas as pd
from evaluate import load
from spoiler_generation.spoiler_generation.utils.dataset_class import Dataset


bleu = load("bleu")


def prepare_data_from_hf(test_path: str, output_path: str) -> pd.DataFrame:
    test_df = pd.read_json(test_path)
    output_df = pd.read_json(output_path, lines=True)
    test_df["answers"] = test_df["answers"].apply(
        lambda x: "\n".join([record["text"][0] for record in x])
    )
    data = []
    for reference, context, question, hypothesis in zip(
        test_df["answers"],
        test_df["context"],
        test_df["question"],
        output_df["spoiler"],
    ):
        data.append(
            [
                question,
                context,
                hypothesis,
                bleu.compute(
                    predictions=[hypothesis],
                    references=[[reference]],
                    max_order=min(4, len(reference.split(" "))),
                )["bleu"],
            ]
        )
    return pd.DataFrame(data, columns=["clickbait", "context", "spoiler", "bleu"])


def prepare_data_from_llms(test_path: str, output_path: str) -> pd.DataFrame:
    test_df = pd.read_json(test_path)
    output_df = pd.read_csv(output_path)
    test_df["output"] = test_df["output"].apply(Dataset.preprocess_func)
    output_df["spoiler"] = output_df["spoiler"].apply(Dataset.preprocess_func)
    output_df["spoiler"].fillna("", inplace=True)
    data = []
    for reference, context, question, hypothesis in zip(
        test_df["output"],
        test_df["context"],
        test_df["question"],
        output_df["spoiler"],
    ):
        if hypothesis == "":
            bleu_value = 0
        else:
            bleu_value = bleu.compute(
                predictions=[hypothesis],
                references=[[reference]],
                max_order=min(4, len(reference.split(" "))),
            )["bleu"]
        data.append([question, context, hypothesis, bleu_value])
    return pd.DataFrame(data, columns=["clickbait", "context", "spoiler", "bleu"])


regressor_df = prepare_data_from_llms(
    "data/llama_generation/validation.json",
    "models/llama-13b-finetuned/val_output.csv",
)
regressor_df.to_csv(
    "spoiler_generation/regressor/best_models_val_data/regressor_data.csv",
    index=False,
    mode="a+",
    header=False,
)
