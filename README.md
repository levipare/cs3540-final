## Setup
Requires
- [uv](https://docs.astral.sh/uv/)

The following creates a venv and installs the required dependencies.
```
uv sync
```

## Using jupyter
This command will run jupyter with access to the venv.
```
uv run --with jupyter jupyter lab
```
