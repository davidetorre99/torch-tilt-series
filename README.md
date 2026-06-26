# torch-tilt-series

[![License](https://img.shields.io/pypi/l/torch-tilt-series.svg?color=green)](https://github.com/teamtomo/torch-tilt-series/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tilt-series.svg?color=green)](https://pypi.org/project/torch-tilt-series)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tilt-series.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-tilt-series/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-tilt-series/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-tilt-series/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-tilt-series)

tilt series

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork teamtomo/torch-tilt-series --clone
# or just
# gh repo clone teamtomo/torch-tilt-series
cd torch-tilt-series
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
