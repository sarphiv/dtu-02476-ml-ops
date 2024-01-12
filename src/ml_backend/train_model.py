from typing import Literal
from pathlib import Path
import os

from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from omegaconf import OmegaConf
import hydra
import yaml

from ml_backend.models.model import BaseModel
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
    pl.seed_everything(cfg.seed)

    ### This will likely be changed in a future version to
    ### enable the choice between multiple models

    # instantiate the pl model
    model = BaseModel(
        model_type=cfg.models.model_type,
        learning_rate=cfg.models.learning_rate,
        weight_decay=cfg.models.weight_decay,
        pretrained=True,
        num_classes=10
    )

    # construct dataloaders
    transform = get_transform(model.model)
    train_dataloader = get_dataloader(transform, "train", batch_size=cfg.models.batch_size, num_workers=cfg.num_workers)
    test_dataloader = get_dataloader(transform, "test", batch_size=cfg.models.batch_size, num_workers=cfg.num_workers)


    # instantiate the logger
    logger = WandbLogger(
        project="dtu_mlops_02476",
        log_model=True,
        entity="metrics_logger",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    # Save the n best checkpoints
    dirpath = Path(cfg.models.model_dir) / logger.experiment.id
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}',
                                          dirpath=dirpath,
                                          save_top_k=cfg.models.save_top_k,
                                          monitor="val_loss")

    # instantiate the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.log_interval
    )

    # train the model
    trainer.fit(model, train_dataloader, test_dataloader)

    # Save the path to the best checkpoint
    checkpoint_callback.to_yaml(dirpath / "best_models.yaml")

    # Compare it with the best run made in this training session
    current_best_path, current_best_loss = min(checkpoint_callback.best_k_models.items(), key=lambda x: x[1].item())
    current_best_loss = current_best_loss.item()

    # If the best model is better than the previous best, save it instead
    with open(Path(cfg.models.model_dir) / "best_model.yaml", "w") as file:
        # Check if the file already exists
        if os.stat(file.name).st_size == 0:
            # If not, save the current best
            global_best = {current_best_path: current_best_loss}
            yaml.dump(global_best, file)
        else:
            # Load the global best run so far
            global_best = yaml.safe_load(file)

            # If this run is better, then save it as the global best
            if list(global_best.values())[0] > current_best_loss:
                global_best = {current_best_path: current_best_loss}
                yaml.dump(global_best, file)



if __name__ == "__main__":
    train()
