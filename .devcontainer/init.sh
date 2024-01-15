#!/bin/bash
pip install -e .[dev]
git config --global --add safe.directory /workspace
pre-commit install
