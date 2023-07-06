export CUDA_VISIBLE_DEVICES="0"
python spoiler_generation/hf_qa_train.py \
--train-data train.json \
--val-data validation.json \
--model-name models/deberta-base_concatenated-v2 \
--output-dir models/not-multi-deberta-v2 \
--batch-size 2 \
--epochs 10 \
--learning-rate 2e-6 \
--max-length 512

