#!/bin/bash

# Development setup script for OpenWeights

echo "Setting up OpenWeights development environment..."

# Install package in editable mode with dev dependencies
echo "Installing package and dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to ensure everything is formatted correctly
echo "Running initial code formatting..."
pre-commit run --all-files

echo "✅ Development environment setup complete!"
echo ""
echo "Pre-commit hooks are now installed and will run automatically before each commit."
echo "You can run 'pre-commit run --all-files' manually to format all files."
