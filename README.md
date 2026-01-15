# mlops

## Project Describtion

### Overall goal
Classifing playing cards into suit and rank.


### What framework are you going to use, and you do you intend to include the framework into your project?
PyTorch

### What data are you going to run on (initially, may change)
The dataset consists of images of playing cards, the dataset can be found here: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/code

### What models do you expect to use
We expect to use MobileNet.


## Project structure
````markdown

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
````


### How to use:

## Train and test the model
- uvx invoke preprocess-data
- uvx invoke train
- uvx invoke evaluate

## run tests to check model
- uvx invoke test



## Enable pre-commit
`uv run pre-commit install`

to ignore pre-commit use `--no-verify` flag when committing, e.g.
`git commit -m <message> --no-verify`

to run precommit manually use
`uv run pre-commit run --all-files`


## Docker

Requieres the wand API key to be in .env.



Build and run train.dockerfile:
- docker build -f dockerfiles/train.dockerfile . -t train:latest
- docker run --env-file .env --name experiment-mlops-train train:latest
OR with envoke:
- uvx invoke docker_build

Build and run evaluate.dockerfile:
- docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
- docker run --env-file .env --name experiment-mlops-evaluate evaluate:latest
