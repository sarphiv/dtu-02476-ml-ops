import torch
import torchvision 

def load_dataset():
    """Loads the raw CIFAR10 dataset (either from "data/raw" or the internet) and saves it in "data/processed" folder"""
    # Load / download the datasets and store them in "data/raw" folder 
    train_dataset = torchvision.datasets.CIFAR10(root="./data/raw", train=True, download=True) 
    test_dataset = torchvision.datasets.CIFAR10(root="./data/raw", train=False, download=True)

    # Create a transform to make all images tensors 
    transform = torchvision.transforms.ToTensor()
    # If we want more transformations, we can add them here
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     ]) 
    train_dataset.transform = transform
    test_dataset.transform = transform

    # Save the datasets in "data/processed" folder
    torch.save(train_dataset, "./data/processed/train_dataset.pt")
    torch.save(test_dataset, "./data/processed/test_dataset.pt")


if __name__ == '__main__':
    # Get the data and process it
    load_dataset()