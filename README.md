# DTU Course - Machine Learning Operations (02476)
This project is part of the DTU course 02476 - Machine Learning Operations.


## Overall goals
Create a website where users can upload pictures, which are then classified by an image classification model, and the result is shown to the user. The infrastructure behind this will consist of containerized services deployed to the cloud with continuous integration and deployment.

The project will focus on the machine learning operation aspects:
- Development happens inside `devcontainers` and will follow the conventional open source development workflow via `Github Flow`, issues, pull requests, and a Kanban board with some continuous integration via `pre-commit`, `ruff`, `PyTest`, and `GitHub Actions` on top. This is to simulate a real life open source project where contributors have to deal with challenges such as different environments and planning.
- `PyTorch Lightning` is used to structure our project when it comes to logging, experiment tracking, and integration of `timm` for finetuning a pretrained `ResNet-18` model. Training experiments are tracked with `Weights & Biases` and `hydra` for experiments and configuration management. Trained models and their associated training data is tracked with `DVC` stored on `GCS Buckets`. Continuous deployment happens through `Cloud Build` to `GCP`, where two `Docker` images are built and deployed to `Cloud Run` - one for the inference backend, another for the website frontend.

## Frameworks
There is a requirement to use a package not covered in the DTU course. This package is PyTorch Image Models (`timm`). It will be used to provide pretrained models, help with preprocessing, and inference via the models.

## Data
The dataset will be `CIFAR10` for fine-tuning. This dataset was chosen simply because of its relatively small size to enable fast development iteration, because the data science aspects of this project are not the focus.

## Models
The initial model will be based upon `ResNet-18`. This model was chosen for its small size so that all members of the group could run the model locally for development purposes without requiring a GPU nor a strong computer. This assumption turned out to be wrong, so there is also a simple MLP model that some members of the group used for development purposes.


# Installation for development
There are two ways to get started with the project. The easy way is with devcontainers, which ensure a consistent environment. The classical approach is to install the project via the `pyproject.toml` file.

1. Approach 1: Devcontainer (recommended)
    1. Install `Docker` (and NVIDIA's container runtime if applicable)
    2. Build and open the appropriate devcontainer (cpu vs. gpu-accelerated)
    3. Run `dvc pull`
2. Approach 2: Pip
    1. Run `pip install .[dev]`
    3. Run `dvc pull`


# Running the project
The project has been deployed as [a frontend server](https://website-server-ym6t3dqyaq-ew.a.run.app/), and [a backend server](https://inference-server-ym6t3dqyaq-ew.a.run.app/docs). Click the links to access them.

To run the project **locally**, you need to do quite a few things.
1. You must be added as a member of the GCP project
2. You must authenticate your DVC with the GCS bucket
4. You must retrieve service account credentials and save to a key file.
3. You must authenticate your WandB installation
5. You may now run the various modules either directly or via Docker.


# Repository of packages structure

The directory structure of the project looks like the following, where only the most important files are shown:

```txt

├── README.md                     <- The top-level README for developers using this repository.
│
├── .devcontainer                 <- Devcontainer configuration
│
├── .dvc                          <- Configuration for DVC
|
├── models.dvc                    <- Tracking of current model version
|
├── data.dvc                      <- Tracking of current data version
│
├── .pre-commit-config.yaml       <- Configuration of pre-commit
|
├── .github                       <- Workflows for GitHub Actions
|
├── cloudbuilds                   <- GCP Cloud Build configurations
|
├── dockerfiles                   <- Dockerfiles for development and deployment
|
├── configs
│   ├── training.models           <- Model specific configuration
│   ├── config.yaml               <- General configuration
│   └── sweep.yaml                <- Configuration of WandB sweep
|
├── data
│   ├── processed                 <- The final, canonical data sets for modeling.
│   └── raw                       <- The original, immutable data dump.
│
├── models                        <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml                <- Project configuration file
│
├── reports                       <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                   <- Generated graphics and figures to be used in reporting
│
├── requirements.txt              <- The requirements file for reproducing the deployment environment
|
├── requirements_website.txt      <- The requirements file for reproducing the website front end environment
|
├── requirements_dev.txt          <- The requirements file for reproducing the development environment
|
|── src                           <- Source code for use in this repository
│   ├── ml_backend                <- Source code for machine learning backend.
│   |   │
│   |   ├── data                  <- Scripts to download or generate data
│   |   |   ├── tests             <- Tests for data handling
│   |   │   ├── __init__.py
│   |   │   └── make_dataset.py
│   |   │
│   |   ├── models                <- model implementations, training script and prediction script
│   |   │
│   |   ├── train_model.py        <- Script for training a model
│   |   ├── inference_server.py   <- Script for running a model inference server
│   |   └── predict.py            <- Script for predicting from a model
|   |
│   └── website                   <- Source code for website front end
│       │
│       └── main.py               <- Script for serving the website
│
└── LICENSE                       <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
