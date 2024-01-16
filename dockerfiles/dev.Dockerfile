# Setup environment basics
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime


# Install packages
RUN apt update -y \
    && apt install -y sudo \
    && apt clean


# Set up user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME


# Set up working directory
WORKDIR /workspace

# Set up environment variables
ENV PYTHONUNBUFFERED=True

# Set up environment
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
