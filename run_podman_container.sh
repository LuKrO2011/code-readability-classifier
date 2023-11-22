#!/bin/bash
#SBATCH --partition=anywhere    # Specify the partition
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1     # Number of tasks per node
#SBATCH --time=1:00:00          # Estimated job duration (hours:minutes:seconds)
#SBATCH --job-name=training_readability_classifier  # Job name
#SBATCH --output=container_output.txt           # Output file for stdout
#SBATCH --error=container_error.txt             # Error file for stderr

# Set the environment variables
unset XDG_RUNTIME_DIR XDG_CONFIG_HOME
export HOME=/local/$USER/podman.home

# Run the specific task/command within the Podman container
podman --root=/local/$USER/podman-compose -f docker-compose.yml run train
