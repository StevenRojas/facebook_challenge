import os
import json


def main():
    path = "data/nets/"
    config_files = [file for file in os.listdir(path) if file.endswith(".json")]
    for file in config_files[::-1]:  # Invert list
        process_config(path, file)


def process_config(path, filename):
    with open(path + filename, 'r') as f:
        config = json.load(f)
        config["filename"] = filename
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
