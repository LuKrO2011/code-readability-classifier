#!/bin/bash

# Set the paths to the program, data, and save directories
PROGRAM_PATH=src/readability_classifier/main.py
DATA_PATH=res/datasets/combined
ENCODED_PATH=res/datasets/encoded
SAVE_PATH=res/models

# Print the paths
echo "Program path: $PROGRAM_PATH"
echo "Data path: $DATA_PATH"
echo "Encoded path: $ENCODED_PATH"
echo "Save path: $SAVE_PATH"

# Run the main program
python "${PROGRAM_PATH}" ENCODE -i "${DATA_PATH}" -s "${SAVE_PATH}" --intermediate "${ENCODED_PATH}"
