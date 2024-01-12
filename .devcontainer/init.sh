#!/bin/bash
pip install -e .[dev]
git config --global --add safe.directory /workspace
git config devcontainers-theme.show-dirty 1
pre-commit install
