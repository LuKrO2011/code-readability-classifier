#!/bin/bash

podman run --rm \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    -v ./src:/app/src \
    -v ./tests/res/models:/app/models \
    -v ./tests/res/code_snippets:/app/snippets \
    rc-gpu \
    python src/readability_classifier/main.py PREDICT \
    --model models/towards.keras \
    --input snippets/towards.java
