#!/bin/bash
#SBATCH --partition=anywhere    # Specify the partition
#SBATCH --constraint=cayman     # Specify the cluster
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1     # Number of tasks per node
#SBATCH --job-name=training_readability_classifier  # Job name
#SBATCH --output=container_output.txt           # Output file for stdout
#SBATCH --error=container_error.txt             # Error file for stderr
#SBATCH --mail-user=krodinger@fim.uni-passau.de # Email

# Set the environment variables
unset XDG_RUNTIME_DIR XDG_CONFIG_HOME
export HOME=/local/$USER/podman.home

# Run the specific task/command within the Podman container
# podman --root=/local/$USER/podman-compose -f docker-compose.yml run train
podman --root=/local/$USER/podman-compose run localhost/rc:4 python src/readability_classifier/main.py TRAIN -i res/datasets/combined -s res/models
