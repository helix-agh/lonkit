# Installation

## Requirements

lonpy requires Python 3.10 or higher.

## Install from PyPI

The simplest way to install lonpy is via pip:

```bash
pip install lonpy
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/helix-agh/lonpy.git
cd lonpy
pip install -e .
```

## Development Installation

To install lonpy with development dependencies for contributing:

```bash
git clone https://github.com/helix-agh/lonpy.git
cd lonpy
pip install -e ".[dev]"
```

This includes:

- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `pre-commit` - Git hooks

## Verifying Installation

After installation, verify everything works:

```python
import lonpy
print(lonpy.__version__)
```

You should see the version number (e.g., `0.1.0`).

## Troubleshooting

### igraph Installation Issues

On some systems, you may need to install igraph system dependencies first:

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get install build-essential python3-dev libxml2-dev zlib1g-dev
    pip install lonpy
    ```

=== "macOS"

    ```bash
    brew install igraph
    pip install lonpy
    ```

=== "Windows"

    igraph typically installs without issues on Windows via pip.

### Plotly/Kaleido Issues

If you encounter issues with static image export:

```bash
pip install --upgrade kaleido
```

On some systems, you may need:

```bash
pip install kaleido==0.2.1
```
