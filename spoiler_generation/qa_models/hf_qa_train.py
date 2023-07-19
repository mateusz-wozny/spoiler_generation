from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)

from spoiler_generation.utils.helpers import create_argparse


def prepare_data(args):
    clickbaits = load_dataset(
        "json",
        data_files={
            "train": args.train_data,
            "eval": args.val_data,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
    )

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=args.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i][0]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return tokenizer, clickbaits.map(
        preprocess_function,
        batched=True,
        remove_columns=clickbaits["train"].column_names,
    )


def train(args, tokenizer, tokenized_clickbaits):
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=5,
        logging_steps=200,
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_clickbaits["train"],
        eval_dataset=tokenized_clickbaits["eval"],
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer


def main(args):
    tokenizer, tokenized_clickbaits = prepare_data(args)

    trainer = train(args, tokenizer, tokenized_clickbaits)
    trainer.save_model()


if __name__ == "__main__":
    parser = create_argparse()
    args = parser.parse_args()
    main(args)
