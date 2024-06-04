#!/bin/bash

podman run --rm \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    -v ../src:/app/src \
    -v ../tests/res/raw_datasets/combined:/app/res \
    -v ../output:/app/output \
    docker.io/lukro2011/rc-gpu:latest \
    python src/readability_classifier/main.py TRAIN \
    -i res \
    -s output

