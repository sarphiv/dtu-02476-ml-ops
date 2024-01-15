#!/bin/fish
echo 'Setting up terminal...'
cp .devcontainer/config.fish ~/.config/fish/config.fish
echo 'y' | fish_config theme save 'ayu Dark'
echo 'y' | fish_config prompt save astronaut

echo 'Setting up development packages...'
pip install -e .[dev]
