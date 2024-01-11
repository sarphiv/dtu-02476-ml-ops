from typing import List, Tuple, Dict, Union, Optional, Callable, Any, Iterable, Literal



import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import urllib.request
from PIL import Image
import matplotlib.pyplot as plt

from ml_backend.models.model import BaseModel


def get_transform(model):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform

def get_dataloader(transform, split: Literal["train", "test"], batch_size: int, num_workers: int, **dataloader_kwargs) -> DataLoader:
    dataset = torch.load(f"./data/processed/CIFAR10/{split}_dataset.pt")
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **dataloader_kwargs)
    return dataloader




# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)
# img = Image.open(filename).convert('RGB')
# tensor = transform(img).unsqueeze(0) # transform and add batch dimension

# with torch.no_grad():
#     out = model(tensor)
#     prob = F.softmax(out, dim=1)[0] * 100
#     print(prob)
#     _, indices = torch.sort(out, descending=True)
#     print([(model.get_classifier().fc.out_features[idx], prob[idx].item()) for idx in indices[0][:5]])



timm_model = timm.create_model('resnet18', pretrained=True, num_classes=10, )


transform = get_transform(timm_model)
train_dataloader = get_dataloader(transform, "train", batch_size=32, num_workers=10)
test_dataloader = get_dataloader(transform, "test", batch_size=32, num_workers=10)



model = BaseModel(timm_model, learning_rate=1e-3)

trainer = pl.Trainer(max_epochs=2, logger=True, log_every_n_steps=1)
trainer.fit(model, train_dataloader, test_dataloader)


optimizer = model.configure_optimizers()

