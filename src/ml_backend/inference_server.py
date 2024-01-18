import io
from PIL import Image
from http import HTTPStatus
import os

from hydra import initialize, compose
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ml_backend.predict import predict_data_class, load_model_best
from ml_backend.storage import load_from_bucket
from ml_backend.ml_logging import MLLogger


with initialize(config_path=os.environ.get("CONFIG_DIR", "../../configs"), version_base="1.3"):
    cfg = compose(config_name="config")
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if cfg.system.load_model_from is not None:
        # Create the directory and load the <model_type> folder into it
        # os.makedirs("models", exist_ok=True)
        load_from_bucket(cfg.system.load_model_from, cfg.system.file_name, "models")

    # Instantiate the model
    model = load_model_best(cfg)



@app.post("/predict/")
async def predict(data: UploadFile = File(...)):
    logger = MLLogger.get_logger()
    # Attempt converting the image to PIL format
    try:
        logger.info("Attempting to load image")
        image = Image.open(io.BytesIO(data.file.read()))
    except Exception:
        logger.error("Invalid image data provided")
        return {
            "message": "Invalid image data provided",
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    logger.info("Image loaded successfully, trying to predict...")
    # Respond with the prediction
    return {
        "predicted-class": predict_data_class(model, image, cfg),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
