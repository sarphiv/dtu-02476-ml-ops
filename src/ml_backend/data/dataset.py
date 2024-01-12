from typing import Callable
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
from torchvision.transforms.functional import to_tensor

from ml_backend.types import channels, height, width


T_transforms = Callable[[Image.Image], TensorType[3, 32, 32]]


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        targets_path: str | Path,
        transform: T_transforms
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path (str or Path): The path to the data file.
            targets_path (str or Path): The path to the targets file.
            transform (callable): A function that applies transformations to the input data.

        Returns:
            None
        """
        super().__init__()

        self.data = torch.load(data_path)
        self.targets = torch.load(targets_path)
        self.transform = transform or to_tensor


    def __getitem__(self, index: int) -> tuple[TensorType[channels, height, width], TensorType[1]]:
        """
        Retrieve the item at the given index from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[Image[3, 32, 32], TensorType[1]]: A tuple containing the transformed data and the target label.
        """
        return (self.transform(Image.fromarray(self.data[index])), self.targets[index])


    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)


def load_dataset(data_dir: str | Path = ".", transform: T_transforms = None) -> tuple[CIFAR10Dataset, CIFAR10Dataset]:
    """Loads the processed CIFAR10 dataset from "data/processed" folder

    Args:
        data_dir (str | Path, optional): The path to the root directory of the project. Defaults to ".".

    Returns:
        tuple[TensorDataset, TensorDataset]: The train and test datasets
    """
    # Load the data
    train_data = Path(data_dir) / "data/processed/CIFAR10/train_dataset.pt"
    test_data = Path(data_dir) / "data/processed/CIFAR10/test_dataset.pt"
    train_targets = Path(data_dir) / "data/processed/CIFAR10/train_targets.pt"
    test_targets = Path(data_dir) / "data/processed/CIFAR10/test_targets.pt"

    # Format into dataset
    train_dataset = CIFAR10Dataset(data_path=train_data, targets_path=train_targets, transform=transform)
    test_dataset = CIFAR10Dataset(data_path=test_data, targets_path=test_targets, transform=transform)

    return train_dataset, test_dataset
