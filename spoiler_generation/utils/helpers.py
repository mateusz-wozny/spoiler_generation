import argparse


def create_argparse():
    parser = argparse.ArgumentParser(description="Run training with parameters")

    # Adding Argument
    parser.add_argument("--train-data", type=str, help="Path to training data", required=True)
    parser.add_argument("--val-data", type=str, help="Path to validation data", required=True)
    parser.add_argument("--batch-size", type=int, help="Batch size", required=True)
    parser.add_argument("--epochs", type=int, help="Num epochs", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", required=True)
    parser.add_argument("--max-length", type=int, help="Max sequence length", required=True)
    parser.add_argument("--model-name", type=str, help="Name of model", required=True)
    parser.add_argument("--output-dir", type=str, help="Output dir for model", required=True)
    return parser
