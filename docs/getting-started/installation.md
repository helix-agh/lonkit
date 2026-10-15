---
description: Installation of lonkit Python library via pip or from source.
---

# Installation

## Requirements

lonkit requires Python 3.10 or higher.

## Install from PyPI

The simplest way to install lonkit is via pip:

```bash
pip install lonkit
```

### Chrome for 3D image and GIF export

Saving 3D plots as images, creating rotation GIFs, and calling `visualize_all()`
require Chrome or Chromium. Kaleido 1.x uses an external browser; it does not
bundle one. If Chrome is already installed, Kaleido should find it automatically.
Otherwise, install a compatible browser with:

```bash
kaleido_get_chrome
```

For a source checkout managed with uv, use `uv run kaleido_get_chrome`.
The download requires an internet connection. In CI, run this command before
the visualization tests. 2D plots and creating Plotly figures without image
export do not require Chrome.

See the [Kaleido installation instructions](https://github.com/plotly/Kaleido#installation)
for details.

## Install from Source

For the latest development version:

```bash
git clone https://github.com/helix-agh/lonkit.git
cd lonkit
pip install -e .
```

## Development Installation

To install lonkit with development dependencies for contributing:

```bash
git clone https://github.com/helix-agh/lonkit.git
cd lonkit
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
import lonkit
print(lonkit.__version__)
```

You should see the version number (e.g., `0.3.0`).

## Troubleshooting

### igraph Installation Issues

On some systems, you may need to install igraph system dependencies first:

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get install build-essential python3-dev libxml2-dev zlib1g-dev
    pip install lonkit
    ```

=== "macOS"

    ```bash
    brew install igraph
    pip install lonkit
    ```

=== "Windows"

    igraph typically installs without issues on Windows via pip.

### Plotly/Kaleido Issues

lonkit requires Plotly >= 6.1.1 and Kaleido >= 1.0.0. If you encounter a version
compatibility error, upgrade both packages together:

```bash
pip install --upgrade "plotly>=6.1.1" "kaleido>=1.0.0"
```

If export reports that Chrome is missing, install it with:

```bash
kaleido_get_chrome
```

Do not downgrade to Kaleido 0.2.1; it is outside lonkit's supported dependencies.
