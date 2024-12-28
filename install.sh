#!/bin/bash

# Define variables
CONDA_INSTALLER="Anaconda3-2024.10-1-Linux-x86_64.sh"
CONDA_URL="https://repo.anaconda.com/archive/$CONDA_INSTALLER"
REPO_URL="https://github.com/abhi2024vlg/samosa_experiments_2025.git"  # Replace with your repository URL
ENV_YAML="environment.yml"
SCRIPT_PATH="main.py"

# Step 1: Install Anaconda
echo "Downloading Anaconda installer..."
wget $CONDA_URL
chmod +x $CONDA_INSTALLER

echo "Installing Anaconda..."
./$CONDA_INSTALLER

# Initialize Anaconda
export PATH="$HOME/anaconda/bin:$PATH"
echo "Initializing Anaconda..."
conda init