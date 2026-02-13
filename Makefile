setup:
	uv sync
	uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
