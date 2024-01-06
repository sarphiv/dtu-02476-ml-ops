# DTU Course - Machine Learning Operations (02476)
This project is part of the DTU course 02476 - Machine Learning Operations. 


## Overall goals
Create a website where users can upload pictures, which are then classified by an image classification model, and the result is shown to the user. The infrastructure behind this will consist of containerized services deployed to the cloud with continuous integration and deployment. 

The project will focus on the machine learning operation aspects:
- Development happens inside `devcontainers` and will follow the conventional open source development workflow via `Github Flow`, issues, pull requests, and a Kanban board with some continuous integration on top. This is to simulate a real life open source project where contributors have to deal with challenges such as different environments and planning.
- `PyTorch Lightning` is used to structure our project when it comes to logging, experiment tracking,and integration of `timm` for finetuning a pretrained model. Training experiments are tracked with `Weights & Biases` and `hydra` for configuration management. The training starts via continuous deployment through `Github Actions` to `GCP`. The model and its associated training data is tracked with `DVC`. Deployment also happens through continuous deployment of `Docker` images to the same service as above.
- Data drift and deployed model performance will be monitored via tools that we have not yet been taught about in the course TODO

## Frameworks
There is a requirement to use a package not covered in the DTU course. This package is PyTorch Image Models (`timm`). It will be used to provide pretrained models, help with preprocessing, and inference via the models. 

## Data
The initial dataset will be `CIFAR10` for fine-tuning. This may be expanded upon later with a larger dataset for better general model performance. 

## Models
The initial model will be based upon `ResNet18`. This model was chosen for its small size so that all members of the group could run the model locally for development purposes without requiring a GPU nor a strong computer. 


# Installation
TODO


# Repository of packages structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this repository.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
|
|── src                  <- Source code for use in this repository
│   └── ml_backend  <- Source code for one package.
│       │
│       ├── __init__.py      <- Makes folder a Python module
│       │
│       ├── data             <- Scripts to download or generate data
│       │   ├── __init__.py
│       │   └── make_dataset.py
│       │
│       ├── models           <- model implementations, training script and prediction script
│       │   ├── __init__.py
│       │   ├── model.py
│       │
│       ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│       │   ├── __init__.py
│       │   └── visualize.py
│       ├── train_model.py   <- script for training the model
│       └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
