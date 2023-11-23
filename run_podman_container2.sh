#! /usr/bin/env bash
#
#SBATCH --job-name=training_readability_classifier
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00  # two hours
#SBATCH --mail-user=krodinger@fim.uni-passau.de
#SBATCH --constraint=thor
#SBATCH --partition=anywhere
#SBATCH --mem-bind=local
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --mail-type=FAIL,END
#
# A container image can be built locally or on a SLURM workstation node and
# then be exported to an archive using:
#
# podman save TODO-image-tag > IMAGE.tar
#
# After the archive was copied to the /scratch/$USER/... directory this script
# can be used to run SLURM jobs in the container.
# Make sure this script is located somewhere in `/scratch/$USER` and executable.
# Then start the job with `sbatch ./this-script.sh`.

set -eu

unset XDG_RUNTIME_DIR XDG_CONFIG_HOME
export HOME=/local/$USER/podman.home
export TMPDIR=/local/$USER/tmp

mkdir -p "/local/${USER}/podman.home/"

echo "Loading image..."
podman \
    --root="/local/$USER/podman" \
    load \
    -i "/scratch/$USER/rc4.tar"

podman \
    --root="/local/$USER/podman" \
    image ls

podman \
    --root="/local/$USER/podman" \
    run \
    --rm \
    # -v "SOME_PATH/datasets:/datasets" \
    # -v "SOME_PATH/results:/eval/results/" \
    --name "rc-container" \
    "localhost/rc:4" \
    python src/readability_classifier/utils/cuda-checker.py

podman \
    --root="/local/$USER/podman" \
    image prune -a -f
