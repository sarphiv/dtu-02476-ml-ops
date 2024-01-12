# Setup environment basics
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Set up ports and environment variables
# EXPOSE 9090
# ENV PYTHONUNBUFFERED=True

# Install system packages
RUN apt update && apt install -y git

# Set up environment
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# # Copy over application
# COPY example.py example.py

# Run model service example
# CMD python example.py
