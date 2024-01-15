FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Exposing port
EXPOSE 8080

# NOTE: Hydra requires relative paths...
ENV CONFIG_DIR=../../../../../../workspace/configs

# Copy over application
WORKDIR /workspace
COPY src src
COPY configs configs
COPY pyproject.toml .
COPY requirements.txt .

# Install our package (this also installs the dependencies)
RUN pip install .

# Run model service example
CMD uvicorn --port 8080 --host 0.0.0.0 ml_backend.inference_server:app
# Run this by: docker run --shm-size=4gb -p 8080:8080 -v C:/Users/david/Desktop/mlops/dtu-02476-ml-ops/models:/workspace/models --env-file=.env way2:latest
