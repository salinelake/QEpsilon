# QEpsilon Documentation

This directory contains the Sphinx documentation for QEpsilon.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

To build the HTML documentation:

```bash
make html
```

The documentation will be built in `_build/html/`. You can open `_build/html/index.html` in your browser to view it.

### Building PDF Documentation

To build PDF documentation:

```bash
make latexpdf
```

### Live Reload During Development

For live reloading during development:

```bash
make livehtml
```

This will start a local server and automatically rebuild the documentation when files change.

### Serving the Documentation

To serve the built documentation locally:

```bash
make serve
```

Then visit `http://localhost:8000` in your browser.

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `theory.rst` - Theoretical background
- `api/` - API documentation
- `examples/` - Example documentation
- `_static/` - Static files (images, CSS, etc.)
- `_templates/` - Custom Sphinx templates

## ReadTheDocs Configuration

The documentation is configured for ReadTheDocs deployment through:

- `../.readthedocs.yaml` - ReadTheDocs configuration
- `requirements.txt` - Documentation dependencies

## Contributing to Documentation

When adding new content:

1. Follow the existing RST format
2. Add code examples with proper syntax highlighting
3. Include docstrings in the source code for API documentation
4. Test locally before submitting

## Updating API Documentation

The API documentation is automatically generated from docstrings. To update:

1. Ensure your code has proper docstrings
2. Rebuild the documentation with `make html`
3. The `sphinx.ext.autodoc` extension will automatically include new classes and functions 