from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer


model = AutoModelForSequenceClassification.from_pretrained(
    "MateuszW/classifier-distilbert",
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(
    "MateuszW/classifier-distilbert",
    model_max_length=512,
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
output = "models/classifier/distilbert-finetuned"
training_args = TrainingArguments(
    output_dir=output,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=300,
    eval_steps=300,
    max_steps=3000,
    learning_rate=7e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
