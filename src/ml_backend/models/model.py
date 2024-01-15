from typing import Literal
import math

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from torchtyping import TensorType

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from ml_backend.types import batch_size, channels, height, width, num_classes
from ml_backend.models.simple_mlp import create_mlp

T_batch = tuple[TensorType[batch_size, channels, height, width], TensorType[batch_size]]

class BaseModel(pl.LightningModule):
    def __init__(
            self,
            model_type: Literal["resnet18", "simple_mlp"],
            learning_rate: float,
            weight_decay: float,
            model_args: dict,
            idx_to_class: dict,
        ):
        """
        Instantiates a ResNet model from timm library as a pytorch lightning module.

        Parameters:
        -----------
        `model_type`: `str`
            the type of the model to be used
        `learning_rate`: `float`
            learning rate to be used for the optimizer
        `weight_decay`: `float`
            weight decay to be used for the optimizer
        model_args: `dict`:
            `pretrained`: `bool`
                whether to use the pretrained model or not (only for timm models)
            `num_classes`: `int`
                number of classes in the dataset
            `hidden_layers`: `int`
                number of hidden layers in the model (only for simple MLP)
            `hidden_dim`: `int`
                hidden dimension of the model (only for simple MLP)
            `input_dim`: `int`
                input dimension of the model (only for simple MLP)
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.model_args = model_args
        self.model = self.create_nn_model(model_type, model_args)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.idx_to_class = idx_to_class

    def get_transform(self):
        """
        get the transform that is used to preprocess the data for the model
        created using the timm module

        Returns:
        --------
        `torchvision.transforms` object
        """
        match self.model_type:
            case "resnet18":
                config = resolve_data_config({}, model=self.model)
                transform = create_transform(**config)

            case "simple_mlp":
                h,w = self.model_args["input_img_height"], self.model_args["input_img_width"]
                transform = transforms.Compose([
                    transforms.Resize(size=math.ceil(235/224*max(h,w)), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias="warn"),
                    transforms.CenterCrop(size=(h,w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
                ])

            case _:
                raise ValueError(f"Model type {self.model_type} is not supported")
            
        return transform

    def create_nn_model(self, model_type: str, model_hypers) -> nn.Module:
        """
        Creates a neural network model from timm library

        Parameters:
        -----------
        `model_type`: `str`
            the type of the model to be used
        `model_hypers`: `dict`
            other keyword arguments to be passed to the model

        Returns:
        --------
        `torch.nn.Module` object
        """
        match model_type:
            case "resnet18":
                return timm.create_model(
                    "resnet18",
                    pretrained=model_hypers["pretrained"],
                    num_classes=10,
                    )
            
            case "simple_mlp":
                return create_mlp(
                    input_dim=model_hypers["input_img_height"] * model_hypers["input_img_width"] * 3,
                    output_dim=10,
                    hidden_dim=model_hypers["hidden_dim"],
                    hidden_layers=model_hypers["hidden_layers"],
                )
            
            case _:
                raise ValueError(f"Model type {model_type} is not supported")

    def forward(self, x: TensorType[batch_size, channels, height, width]) -> TensorType[batch_size, num_classes]:
        """
        Forward pass of the model

        Parameters:
        -----------
        `x`: `torch.Tensor` of shape `(batch_size, channels, height, width)`
            Input tensor to the model
        """
        return self.model(x)


    def _step_helper(self, batch: T_batch, batch_idx: int, mode: str) -> TensorType[1]:
        """
        Makes a forward pass, calculates loss and logs it
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean().item()
        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_accuracy", accuracy)
        return loss


    def training_step(self, batch: T_batch, batch_idx: int) -> TensorType[1]:
        return self._step_helper(batch, batch_idx, "train")


    def validation_step(self, batch: T_batch, batch_idx: int) -> TensorType[1]:
        return self._step_helper(batch, batch_idx, "val")


    def test_step(self, batch: T_batch, batch_idx: int) -> TensorType[1]:
        return self._step_helper(batch, batch_idx, "test")


    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
