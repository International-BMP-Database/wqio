## Setting up a development environment

`wqio` uses [uv](https://docs.astral.sh/uv/) to manage its environment, dependencies, and builds.
Python 3.12 and newer are supported.

```shell
git clone git@github.com:International-BMP-Database/wqio.git
cd wqio
uv sync
```

This creates `.venv/` and installs `wqio` (in editable mode) along with the `dev` dependency group.
Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; commit changes to both together, e.g., after `uv add <package>` or `uv add --dev <package>`.

## Running the tests

```shell
uv run python check_wqio.py                  # basic tests
uv run python check_wqio.py --strict         # also runs doctests and image comparison tests
```

## Building the docs

The documentation dependencies are in the `docs` dependency group:

```shell
uv sync --group docs
cd docs
uv run sphinx-build -b html . _build/html
```

## Code style

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/) (configured in `ruff.toml`), and imports are sorted with isort.
To run these automatically before each commit:

```shell
uv run pre-commit install
```

Or run them by hand:

```shell
uv run ruff check --fix .
uv run ruff format .
```

For anything not covered by the tools, please refer to matplotlib's [Coding Guidelines](https://matplotlib.org/devel/coding_guide.html).

## Git workflow
Please refer to matplotlib's [Git Workflow](https://matplotlib.org/devel/gitwash/development_workflow.html).

## Building

```shell
uv build
```

This writes an sdist and a wheel to `dist/`.
See the "Releases" section of the readme for how releases are made.
