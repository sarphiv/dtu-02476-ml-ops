import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import typing

import timm

class BaseModel(pl.LightningModule):
    def __init__(self, timm_model, learning_rate):
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
        self.save_hyperparameters()
        self.model = timm_model
        self.learning_rate = learning_rate
    
    def forward(self, x):
        """
        Forward pass of the model

        Parameters:
        -----------
        `x`: `torch.Tensor` of shape `(batch_size, channels, height, width)`
            Input tensor to the model
        """
        return self.model(x)
    
    def _step_helper(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(f'{mode}_loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    