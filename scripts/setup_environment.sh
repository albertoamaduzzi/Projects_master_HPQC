#!/bin/bash

# Define the environment name
ENV_NAME="my_science_env"

# Load Python module (adjust version based on your cluster's availability)
module load python

# Create the virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    python3 -m venv $ENV_NAME
    echo "Virtual environment created."
fi

# Activate the environment
source $ENV_NAME/bin/activate

# Upgrade pip and install packages
pip install --upgrade pip
pip install numpy matplotlib scipy pandas seaborn

echo "Installation complete."