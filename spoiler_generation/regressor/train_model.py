from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "MateuszW/regressor-deberta-iter1-iter2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(
    "MateuszW/regressor-deberta-iter1-iter2",
    model_max_length=512,
)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_dataset = load_dataset(
    "MateuszW/spoiler_generation",
    data_files={
        "train": "regressor_data/train.csv",
        "validation": "regressor_data/validation.csv",
    },
).map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("bleu", "label")

output = "spoiler_generation/regressor/best-model-deberta-finetune-v2"
training_args = TrainingArguments(
    output_dir=output,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=300,
    eval_steps=300,
    max_steps=1200,
    learning_rate=2e-6,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    weight_decay=0.01,
    logging_steps=100,
    push_to_hub=False,
    load_best_model_at_end=True,
    report_to="mlflow",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output)
