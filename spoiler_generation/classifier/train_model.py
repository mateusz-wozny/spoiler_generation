from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from spoiler_generation.utils.helpers import create_argparse


def prepare_data(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = load_dataset(
        "MateuszW/spoiler_generation",
        data_files={
            "train": "clf_data/train.csv",
            "validation": "clf_data/val.csv",
        },
    ).map(tokenize_function, batched=True)
    return tokenizer, tokenized_dataset


def train(args, tokenizer, tokenized_clickbaits):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_steps=100,
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="mlflow",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_clickbaits["train"],
        eval_dataset=tokenized_clickbaits["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer


def main(args):
    tokenizer, tokenized_clickbaits = prepare_data(args)

    trainer = train(args, tokenizer, tokenized_clickbaits)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = create_argparse()
    args = parser.parse_args()
    main(args)
