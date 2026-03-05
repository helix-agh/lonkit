# Contributing

Thank you for your interest in contributing to lonpy!

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/helix-agh/lonpy.git
cd lonpy
```

2. Install with development dependencies:

```bash
uv sync --all-extras
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Development Workflow

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=lonpy --cov-report=html
```

### Code Quality

Run all pre-commit hooks:

```bash
pre-commit run --all-files
```

Individual tools:

```bash
# Linting
ruff check src/

# Formatting
ruff format src/

# Type checking
mypy src/
```

### Building Documentation

Preview locally:

```bash
mkdocs serve
```

Build static site:

```bash
mkdocs build
```

## Code Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use type hints for all public functions
- Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep functions focused and small

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with a clear message
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

When reporting bugs, please include:

- Python version
- lonpy version
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

## Feature Requests

Feature requests are welcome! Please:

- Check existing issues first
- Describe the use case
- Explain why it would be useful

## Questions

For questions about using lonpy:

- Check the [documentation](index.md)
- Search existing [issues](https://github.com/helix-agh/lonpy/issues)
- Open a new issue with the "question" label
