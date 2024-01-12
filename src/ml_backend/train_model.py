from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from omegaconf import OmegaConf
import hydra

from ml_backend.models.model import LightningWrapper
from ml_backend.models.simple_mlp import create_mlp
from ml_backend.data.dataset import CIFAR10Dataset




def get_transform(model: nn.Module):
    """
    get the transform that is used to preprocess the data for the model
    created using the timm module

    Parameters:
    ----------
    `model`: `nn.Module`
        the timm model to be used

    Returns:
    --------
    `torchvision.transforms` object
    """
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform

def get_dataloader(transform, split: Literal["train", "test"], batch_size: int, num_workers: int, **dataloader_kwargs) -> DataLoader:
    """
    get the train/test dataloader for the CIFAR10 dataset
    before running this function, make sure that the dataset is downloaded and processed
    (use the ml_backend.data.make_dataset.load_dataset function)

    Parameters:
    -----------
    `transform`: `torchvision.transforms` object
        the transform to be used to preprocess the data
        output from `get_transform` function
    `split`: `str`
        either "train" or "test"
    `batch_size`: `int`
        batch size to be used for the dataloader
    `num_workers`: `int`
        number of workers to be used for the dataloader
    `**dataloader_kwargs`: `dict`
        other keyword arguments to be passed to the dataloader
    """
    # Get the data
    data = f"./data/processed/CIFAR10/{split}_dataset.pt"
    labels = f"./data/processed/CIFAR10/{split}_targets.pt"
    dataset = CIFAR10Dataset(data_path=data, targets_path=labels, transform=transform)
    # Change to a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **dataloader_kwargs)
    return dataloader


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg):
    """
    train the model using the config file and hydra
    """

    # set seed
    pl.seed_everything(cfg.training.seed)

    ### This will likely be changed in a future version to
    ### enable the choice between multiple models

    match cfg.training.models.model_type:
        case "resnet18":

            # load the timm model
            nn_model = timm.create_model('resnet18', pretrained=True, num_classes=10, )

            # construct dataloaders
            transform = get_transform(nn_model)
        
        case "simple_mlp":
            nn_model = create_mlp(
                input_dim=32*32*3,
                output_dim=10,
                hidden_dim=cfg.training.models.hidden_dim,
                hidden_layers=cfg.training.models.hidden_layers
            )

            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=34, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias="warn"),
                torchvision.transforms.CenterCrop(size=(32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])

    
    train_dataloader = get_dataloader(transform, "train", batch_size=cfg.training.models.batch_size, num_workers=cfg.training.num_workers)
    test_dataloader = get_dataloader(transform, "test", batch_size=cfg.training.models.batch_size, num_workers=cfg.training.num_workers)

    # instantiate the pl model
    model = LightningWrapper(
        nn_model,
        learning_rate=cfg.training.models.learning_rate,
        weight_decay=cfg.training.models.weight_decay
    )

    # instantiate the logger
    logger = WandbLogger(
        project=cfg.system.wandb_project,
        log_model=False,
        entity=cfg.system.wandb_entity,
        config=OmegaConf.to_container(cfg.training, resolve=True, throw_on_missing=True),
    )

    # instantiate the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        log_every_n_steps=cfg.training.log_interval
    )

    # train the model
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    train()
