from pathlib import Path

import torch
import torchvision


def make_dataset(data_dir: str | Path = ".") -> None:
    """Loads the raw CIFAR10 dataset (either from "data/raw" or the internet) and saves it in "data/processed" folder

    Args:
        data_dir (str | Path, optional): The path to the root directory of the project. Defaults to ".".
    """
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
    train_data = train_dataset.data.transpose((0, 3, 1, 2))
    test_data = test_dataset.data.transpose((0, 3, 1, 2))
    train_targets = torch.tensor(train_dataset.targets)
    test_targets = torch.tensor(test_dataset.targets)

    # Convert to tensors and normalize
    train_data = torch.from_numpy(train_data).float() / 255
    test_data = torch.from_numpy(test_data).float() / 255

    # Save the data in "data/processed" folder
    torch.save(train_data, processed_path / "train_dataset.pt")
    torch.save(test_data, processed_path / "test_dataset.pt")
    torch.save(train_targets, processed_path / "train_targets.pt")
    torch.save(test_targets, processed_path / "test_targets.pt")


if __name__ == "__main__":
    # Get the data and process it
    make_dataset()
