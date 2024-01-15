#!/bin/fish
# NOTE: Must run this in start to avoid creating a .gitconfig
#  after container creation, which prevents copy of local .gitconfig
echo 'Setting up git...'
pre-commit install
git config --global --add safe.directory /workspace
git config devcontainers-theme.show-dirty 1
