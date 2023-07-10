export CUDA_VISIBLE_DEVICES="0"
python spoiler_generation/hf_qa_train.py \
--train-data train.json \
--val-data validation.json \
--model-name MateuszW/deberta-paqott \
--output-dir models/deberta-finetuned \
--batch-size 2 \
--epochs 10 \
--learning-rate 2e-6 \
--max-length 512

