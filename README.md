## Setup
Requires
- [uv](https://docs.astral.sh/uv/)

The following creates a venv and installs the required dependencies.
```
uv sync
```

## Streamlit Dashboard
Run `make streamlit` to launch the dashboard.

## Using jupyter
This command will run jupyter with access to the venv.
```
uv run --with jupyter jupyter lab
```
