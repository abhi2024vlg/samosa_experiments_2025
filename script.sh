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

# Refresh shell to load Anaconda
source ~/.bashrc

# Step 3: Create Conda environment from YAML
if [ -f "$ENV_YAML" ]; then
    echo "Creating Conda environment from $ENV_YAML..."
    conda env create -f $ENV_YAML
else
    echo "Environment YAML file not found!"
    exit 1
fi

echo "Setting Hugging Face credentials..."
source token.env
huggingface-cli login --token $HF_TOKEN

pip install antlr4-python3-runtime==4.7.1

python3 main.py