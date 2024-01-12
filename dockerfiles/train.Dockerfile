FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Copy over application
WORKDIR /workspace
COPY src src
COPY configs configs
COPY pyproject.toml .
COPY requirements.txt .

# Install our package (this also installs the dependencies)
RUN pip install .

# Run model service example
CMD python src/ml_backend/data/make_dataset.py && python src/ml_backend/train_model.py
