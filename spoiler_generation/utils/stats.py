from evaluate import load
import mlflow
from spoiler_generation.utils.dataset_class import Dataset


bleu = load("bleu")
bertscore = load("bertscore")
exact_match = load("exact_match")
meteor = load("meteor")


def mean(score: list):
    return sum(score) / len(score)


def calculate_exact_match(true_spoiler, predicted_spoiler):
    return exact_match.compute(
        predictions=list(predicted_spoiler["spoiler"].loc[true_spoiler.index]),
        references=list(true_spoiler["spoiler"]),
        ignore_case=True,
        ignore_punctuation=True,
    )


def calculate_bleu(true_spoiler, predicted_spoiler):
    bleu_score = []
    true_spoiler["spoiler"] = true_spoiler["spoiler"].apply(Dataset.preprocess_func)
    predicted_spoiler["spoiler"] = predicted_spoiler["spoiler"].apply(Dataset.preprocess_func)

    for reference, hypothesis in zip(true_spoiler["spoiler"], predicted_spoiler["spoiler"].loc[true_spoiler.index]):
        try:
            val = bleu.compute(
                predictions=[hypothesis],
                references=[[reference]],
                max_order=min(4, len(reference.split(" "))),
            )["bleu"]
        except ZeroDivisionError:
            val = 0
        bleu_score.append(val)
    return mean(bleu_score)


def calculate_bertscore(true_spoiler, predicted_spoiler):
    results = bertscore.compute(
        predictions=list(predicted_spoiler["spoiler"].loc[true_spoiler.index]),
        references=list(true_spoiler["spoiler"]),
        lang="en",
    )
    return results


def bert_metrics_mean(bert_score_val):
    mean_vals = {}
    for name in ["precision", "recall", "f1"]:
        mean_vals[name] = mean(bert_score_val[name])
    return mean_vals


def calculate_meteor(true_spoiler, predicted_spoiler):
    meteor_score = []
    true_spoiler["spoiler"] = true_spoiler["spoiler"].apply(Dataset.preprocess_func)
    predicted_spoiler["spoiler"] = predicted_spoiler["spoiler"].apply(Dataset.preprocess_func)

    for reference, hypothesis in zip(true_spoiler["spoiler"], predicted_spoiler["spoiler"].loc[true_spoiler.index]):
        val = meteor.compute(
            predictions=[hypothesis],
            references=[reference],
        )["meteor"]

        meteor_score.append(val)
    return mean(meteor_score)


def prepare_stats(true_spoiler, predicted_spoiler):
    true_spoiler["spoiler"] = true_spoiler["spoiler"].apply(Dataset.preprocess_func)
    predicted_spoiler["spoiler"] = predicted_spoiler["spoiler"].apply(Dataset.preprocess_func)

    stats = {}
    stats["bleu"] = calculate_bleu(true_spoiler, predicted_spoiler)
    stats.update(bert_metrics_mean(calculate_bertscore(true_spoiler, predicted_spoiler)))
    stats.update(calculate_exact_match(true_spoiler, predicted_spoiler))
    stats["meteor"] = calculate_meteor(true_spoiler, predicted_spoiler)
    return stats


def log_to_mlflow(output_dir, metrics, run_id=None):
    if run_id is None:
        run_id = mlflow.search_runs("0", filter_string=f"params.output_dir='{output_dir}'")["run_id"][0]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
