import typing

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import timm
from torchtyping import TensorType

class BaseModel(pl.LightningModule):
    def __init__(self, timm_model: torch.nn.Module, learning_rate: float):
        """
        Instantiates a ResNet model from timm library as a pytorch lightning module.

        Parameters:
        -----------
        `timm_model`: `timm.models` object
            the timm model to be used
        `learning_rate`: `float`
            learning rate to be used for training
        
        """
        super().__init__()
        self.save_hyperparameters(ignore=['timm_model'])
        self.model = timm_model
        self.learning_rate = learning_rate
    
    def forward(self, x: TensorType["batch_size", "channels", "height", "width"]) -> TensorType["batch_size", "num_classes"]:
        """
        Forward pass of the model

        Parameters:
        -----------
        `x`: `torch.Tensor` of shape `(batch_size, channels, height, width)`
            Input tensor to the model
        """
        return self.model(x)
    
    def _step_helper(self, batch: int, batch_idx: int, mode: str) -> TensorType[1]:
        """
        Makes a forward pass, calculates loss and logs it
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean().item()
        self.log(f'{mode}_loss', loss.item())
        self.log(f'{mode}_accuracy', accuracy)
        return loss
    
    def training_step(self, batch: int, batch_idx: int) -> TensorType[1]:
        return self._step_helper(batch, batch_idx, 'train')
    
    def validation_step(self, batch: int, batch_idx: int) -> TensorType[1]:
        return self._step_helper(batch, batch_idx, 'val')
    
    def test_step(self, batch: int, batch_idx: int) -> TensorType[1]:
        return self._step_helper(batch, batch_idx, 'test')
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    