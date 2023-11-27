#!/bin/bash

# Set the paths to the program, data, and save directories
MODEL=TOWARDS
BATCH_SIZE=8
EPOCHS=20
LEARNING_RATE=0.0015
K_FOLD=10
PROGRAM_PATH=src/readability_classifier/main.py
DATA_PATH=res/datasets/combined
SAVE_PATH=res/models

# Display the provided model
echo "Selected model: $MODEL"
echo "Data path: $DATA_PATH"
echo "Save path: $SAVE_PATH"
echo "Program path: $PROGRAM_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "K fold: $K_FOLD"

# Run the main program
python "${PROGRAM_PATH}" TRAIN -i "${DATA_PATH}" -s "${SAVE_PATH}" -m "${MODEL}" -b "${BATCH_SIZE}" -e "${EPOCHS}" -r "${LEARNING_RATE}" -k "${K_FOLD}"
