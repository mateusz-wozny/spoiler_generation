python spoiler_generation/llms/peft_train.py \
--train-data train.json \
--val-data validation.json \
--model-name facebook/opt-13B \
--output-dir models/llms/opt \
--batch-size 10 \
--epochs 3 \
--learning-rate 1e-5 \
--max-length 1024

