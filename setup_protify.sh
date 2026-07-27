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

# Reinstall the FastPLMs 1.0 validation build from the CUDA wheel index.
echo "Installing PyTorch 2.13..."
pip install --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install --force-reinstall numpy==1.26.4

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

