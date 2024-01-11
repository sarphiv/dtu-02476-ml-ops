import os

import pytest

from ml_backend.data.make_dataset import make_dataset
from ml_backend.data.dataset import load_dataset


@pytest.fixture(scope="session")
def data_directory(tmpdir_factory):
    dir_name = tmpdir_factory.mktemp("data")
    # Run make_dataset.py
    make_dataset(dir_name)
    return dir_name


def test_data_exists(data_directory):
    # Checks if the raw data exists
    assert os.path.exists(data_directory / "/data/raw/CIFAR10/cifar-10-batches-py")
    assert os.path.exists(data_directory / "/data/raw/CIFAR10/cifar-10-python.tar.gz")

    # Checks if the processed data exists
    assert os.path.exists(data_directory / "/data/processed/CIFAR10/train_dataset.pt")
    assert os.path.exists(data_directory / "/data/processed/CIFAR10/test_dataset.pt")


def test_data_dimensions(data_directory): 
    # Load the processed data
    train_dataset, test_dataset = load_dataset(data_directory)

    # Check the dimensions of the loaded data
    assert train_dataset.data.shape == (50000, 3, 32, 32)
    assert test_dataset.data.shape == (10000, 3, 32, 32)
    
    # Check the dimensions of targets
    assert train_dataset.targets.shape == (50000,)
    assert test_dataset.targets.shape == (10000,)

    # Check the dimensions of a single image
    assert train_dataset[0][0].shape == (3, 32, 32)

