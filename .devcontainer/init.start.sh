#!/bin/fish
# NOTE: Must run this in start to avoid creating a .gitconfig
#  after container creation, which prevents copy of local .gitconfig
echo 'Setting up git...'
git config --global --add safe.directory /workspace
git config devcontainers-theme.show-dirty 1
pre-commit install
