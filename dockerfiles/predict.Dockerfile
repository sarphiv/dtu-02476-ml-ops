FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Exposing port
EXPOSE 8080

# Copy over application
WORKDIR /workspace
COPY src .
COPY configs .
COPY pyproject.toml .
COPY requirements.txt .

# Install our package (this also installs the dependencies)
RUN pip install .

# Run model service example
CMD uvicorn --port 8080 ml_backend.inference_server:app
