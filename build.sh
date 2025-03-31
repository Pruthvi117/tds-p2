#!/bin/bash

# Exit on error
set -e

# Print Python version
python --version

# Install pip-tools for better dependency resolution
pip install pip-tools

# Compile requirements
pip-compile requirements.txt -o compiled_requirements.txt

# Install compiled requirements
pip install -r compiled_requirements.txt

echo "Build completed successfully!"
