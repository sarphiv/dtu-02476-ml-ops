import io
from PIL import Image
from http import HTTPStatus
import os

from fastapi import FastAPI, UploadFile, File
from hydra import initialize, compose

from ml_backend.predict import predict_data_class, load_model_best

print(os.getcwd())

# with initialize(config_path="../../configs", version_base="1.3"):
#     cfg = compose(config_name="config")
#     app = FastAPI()
#     model = load_model_best(cfg)

with initialize(config_path=os.environ.get("CONFIG_DIR", "../../configs"), version_base="1.3"):
    print(os.getcwd())
    cfg = compose(config_name="config")
    print(os.getcwd())
    app = FastAPI()
    print(os.getcwd(), cfg.training.models.model_dir)
    model = load_model_best(cfg)



@app.post("/predict/")
async def predict(data: UploadFile = File(...)):
    # Attempt converting the image to PIL format
    try:
        image = Image.open(io.BytesIO(data.file.read()))
    except Exception:
        return {
            "message": "Invalid image data provided",
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    # Respond with the prediction
    return {
        "predicted-class": predict_data_class(model, image, cfg),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
