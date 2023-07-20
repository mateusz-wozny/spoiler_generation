# Spoiler generation

This repository contains code used for training spoiler generation models based on Question Answering, text generation, and learn-to-rank approaches. To use this code please type:

```bash
git clone https://github.com/mateusz-wozny/spoiler_generation.git
cd spoiler_generation
```
Then install required packages by typing:
```bash
pip install -e .
```
It will install all needed libraries and the package in editable mode.

## Code structure
Code available under `spoiler_generation` directory is organized in several directories. Four directories contain a file named `run_training.sh` which allows to run model training for the selected approach.

The specific directories describe selected approaches as follows:
 - `classifier` - contains code used for training pairwise ranking model (i.e. classifier). It is possible to train (or fine-tune) selected models using `run_training.sh` file. In this file, the user has to specify the model name selected from HuggingFace Hub, output directory, and hyperparameters for training i.e. batch size, number of training epochs, learning rate, and max text length. Training and validation data is taken from previously prepared datasets available at the link [pairwise model dataset](https://huggingface.co/datasets/MateuszW/spoiler_generation/tree/main/clf_data). The attached notebook allows to check the quality of the trained model on the validation dataset.
 - `ensemble` - has only one attached notebook which allows previously trained ranking models to be used. There is shown usage of pointwise and pairwise ranking models.
 - `llms` - contains files for training and inference for Large Language Models finetuned using LoRA adapter. Similarly to the pairwise ranking model approach exists file for run training, but the first step is different. First of all, is necessary to prepare the dataset for training and validation. It can be realized using `prepare_dataset.py` by CLI (command-line interface) or using `prepare_dataset.ipynb`. In the first case please enter the below code in the terminal:

    ```bash
    python spoiler_generation/llms/prepare_dataset.py <output_directory>
    ```
    
    It will create two files in `<output_directory>` - `train.json` and `validation.json`. Then you can use `run_training.sh` similarly to the pairwise ranking model. You can also change the used prompt at the top of `peft_train.py` file. For inference was created a notebook `inference.ipynb` in which the first model with adapters is loaded and then for the selected part of the dataset spoilers are generated.
- `qa_models` - likewise in this case also the first step is to prepare the dataset for training. The pattern is the same as the previous one. Also, the start of training is almost identical. On the other hand, for inference was created `predict.sh` file. To run inference you have to pass the path of the dataset file in json format, model name (or model directory for local training), and output file path.To run inference you have to pass the path of the dataset file in json format, model name (or model directory for local training), and output file path.To run inference you have to pass the path of the dataset file in json format, model name (or model directory for local training), and output file path.
- `regressor` - contains code for training, pointwise ranking model. Usage is very similar to the pairwise approach. Training and validation datasets are also previously prepared and available at the link [pointwise model dataset](https://huggingface.co/datasets/MateuszW/spoiler_generation/tree/main/regressor_data)

In addition, separately in `notebooks` directory exists a notebook with a simple analysis of generated spoilers for selected models. This code uses the test part of the dataset which is private.

## Used models
Used models are available at the following links:
 - [pairwise ranking model](https://huggingface.co/MateuszW/classifier-distilbert)
 - [pointwise ranking model ](https://huggingface.co/MateuszW/regressor-deberta-iter1-iter2)
  - [DeBERTa trained on questions](https://huggingface.co/MateuszW/deberta-paqott)
  - [LLaMA adapter](https://huggingface.co/MateuszW/llama-pwt-adapter)  
  - [Vicuna adapter](https://huggingface.co/MateuszW/vicuna-pwt-adapter)
  - [Vicuna type-based prompts adapter](https://huggingface.co/MateuszW/vicuna-ppt-adapter)