export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR="models/not-multi-deberta-v2"
python spoiler_generation/transformers_tutorial/predict_model.py \
--input test.json \
--output $OUTPUT_DIR/output.jsonl \
--model $OUTPUT_DIR