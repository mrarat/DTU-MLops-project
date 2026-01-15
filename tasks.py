import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py models/model.pth", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run('uv run coverage run --source=src --omit="tests/*,/tmp/*" -m pytest tests/', echo=True, pty=not WINDOWS)
    ctx.run('uv run coverage report -m -i --omit="tests/*,/tmp/*"', echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context) -> None:
    """Build docker images."""
    ctx.run(
        "docker build -f dockerfiles/train.dockerfile . -t train:latest",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        "docker run --env-file .env --name experiment-mlops-train train:latest", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
