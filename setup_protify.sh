#!/bin/bash

# chmod +x setup_protify.sh
# ./setup_protify.sh

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Python virtual environment for Protify..."

# Create virtual environment
python3 -m venv ~/protify_venv

# Activate virtual environment
source ~/protify_venv/bin/activate

# Update pip and setuptools
echo "Upgrading pip and setuptools..."
pip install pip setuptools -U

# Install requirements with force reinstall
echo "Installing requirements"
pip install -r requirements.txt -U

# Install torch and torchvision
echo "Installing torch and torchvision..."
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U
pip install --force-reinstall numpy==1.26.4 "fsspec[http]>=2023.1.0,<=2025.10.0"

# Fail immediately if a later reinstall introduced an incompatible dependency.
pip check

# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

# Instructions for future use
echo -e "\n======================="
echo "Setup complete!"
echo "======================="
echo "To activate this environment in the future, run:"
echo "    source ~/protify_venv/bin/activate"
echo ""
echo "To deactivate the environment, simply run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is located at: ~/protify_venv"
echo "======================="

