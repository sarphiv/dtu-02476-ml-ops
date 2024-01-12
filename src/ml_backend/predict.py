from pathlib import Path
from typing import Any
import os
import yaml
from PIL import Image

from torchtyping import TensorType
from torchvision.transforms import Resize
import hydra

from ml_backend.models.model import BaseModel
from ml_backend.types import batch_size, channels, height, width, num_classes
from ml_backend.data.dataset import load_dataset
from ml_backend.train_model import get_transform

def load_model(model_path: str, device: str = "cpu") -> BaseModel:
    """
    load the model from the given path

    Parameters:
    -----------
    `model_path`: `str`
        path to the model or to a txt file storing the path of the model
    `device`: `str`
        device to be used for loading the model

    Returns:
    --------
    `BaseModel` object
    """
    # If the model
    if model_path[-4:] == ".txt":
        model_path = open(model_path, "r").read().strip()

    # Load the model
    model = BaseModel.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    return model

def load_model_best(cfg: Any) -> BaseModel:
    # If file does not exist, then raise error
    if not os.path.isfile(Path(cfg.models.model_dir) / "best_model.yaml"):
        raise FileNotFoundError("best_model.yaml not found. You must train a model first.")

    # Load the best model so far
    with open(Path(cfg.models.model_dir) / "best_model.yaml", "r") as file:
        best_model = yaml.safe_load(file)
        model_path = list(best_model.keys())[0]

    # Load the model
    return load_model(model_path)


def transform_picture_to_data_point(img : Image.Image, cfg: Any) -> TensorType[1, channels, height, width]:
    """
    Resizes image to the correct size
    """
    # Get the image size
    img_size = cfg.models.img_size

    # Resize the image
    resized_img = Resize((img_size, img_size))(img)
    RGB_img = resized_img.convert("RGB")

    return RGB_img


def predict_data_probabilities(
    model: BaseModel,
    data: Image.Image,
    cfg: Any
    ) -> TensorType[batch_size, num_classes]:
    """
    Predict the data using the given model

    Parameters:
    -----------
    `model`: `BaseModel` object
        model to be used for prediction
    `data`: `torch.Tensor` of shape `(batch_size, channels, height, width)`
        data to be predicted

    Returns:
    --------
    `torch.Tensor` of shape `(batch_size, num_classes)`
    """
    # Get the transform for that model
    transform = get_transform(model.model)

    # Resize the image and apply the transform
    data = transform_picture_to_data_point(data, cfg)
    data = transform(data)

    return model(data)


def predict_data_class(
    model: BaseModel,
    data: Image.Image,
    cfg: Any
    ) -> int:
    """
    Predict the data using the given model

    Parameters:
    -----------
    `model`: `BaseModel` object
        model to be used for prediction
    `data`: `torch.Tensor` of shape `(batch_size, channels, height, width)`
        data to be predicted

    Returns:
    --------
    `torch.Tensor` of shape `(batch_size)`
    """
    # Get the transform for that model
    transform = get_transform(model.model)

    # Resize the image and apply the transform
    data = transform_picture_to_data_point(data, cfg)
    data = transform(data).unsqueeze(0)

    return model(data).argmax(1).item()


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg):
    # Load the best model
    model = load_model_best(cfg)

    # Get the data
    transforms = get_transform(model.model)
    _, test_dataset = load_dataset(transform=transforms)

    # Predict
    predictions = model(test_dataset[0][0].unsqueeze(0)).argmax(1).item()
    print(predictions, test_dataset[0][1])


if __name__ == "__main__":
    main()
