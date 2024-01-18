from pathlib import Path
import json

import numpy as np
import torch
import torchvision

from ml_backend.ml_logging import MLLogger

# DATA_TO_USE = 0.01


def make_dataset(data_dir: str | Path = ".") -> None:
    """Loads the raw CIFAR10 dataset (either from "data/raw" or the internet) and saves it in "data/processed" folder

    Args:
        data_dir (str | Path, optional): The path to the root directory of the project. Defaults to ".".
    """
    # Get the logger
    logger = MLLogger()
    logger.INFO("Making the dataset")

    # Create paths
    raw_path = Path(data_dir) / "data/raw/CIFAR10"
    processed_path = Path(data_dir) / "data/processed/CIFAR10"

    # Create the "data/raw/CIFAR10" and "data/processed/CIFAR10" folders if they don't exist
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Load / download the datasets and store them in "data/raw" folder
    train_dataset = torchvision.datasets.CIFAR10(root=raw_path, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=raw_path, train=False, download=True)

    # Get the data
    # NOTE: Nothing is done for now
    # n_points_to_use_train = int(DATA_TO_USE * len(train_dataset))
    # n_points_to_use_test = int(DATA_TO_USE * len(test_dataset))

    # train_data = train_dataset.data[:n_points_to_use_train]
    # test_data = test_dataset.data[:n_points_to_use_test]
    # train_targets = np.array(train_dataset.targets)[:n_points_to_use_train]
    # test_targets = np.array(test_dataset.targets)[:n_points_to_use_test]

    train_data = train_dataset.data
    test_data = test_dataset.data
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    # Save the data in "data/processed" folder
    torch.save(train_data, processed_path / "train_dataset.pt")
    torch.save(test_data, processed_path / "test_dataset.pt")
    torch.save(train_targets, processed_path / "train_targets.pt")
    torch.save(test_targets, processed_path / "test_targets.pt")
    with open(processed_path / "idx_to_class.json", "w") as f:
        json.dump({v:k for k,v in train_dataset.class_to_idx.items()}, f)


if __name__ == "__main__":
    # Get the data and process it
    make_dataset()
