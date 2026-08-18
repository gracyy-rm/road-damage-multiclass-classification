import json
import argparse

from src.train import run_pipeline

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    run_pipeline(config)