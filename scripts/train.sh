#!/bin/bash

# Get the model name from the environment variable
MODEL=$MODEL
DATA_PATH=$DATA_PATH
SAVE_PATH=$SAVE_PATH
PROGRAM_PATH=$PROGRAM_PATH
BATCH_SIZE=$BATCH_SIZE
EPOCHS=$EPOCHS
LEARNING_RATE=$LEARNING_RATE
K_FOLD=$K_FOLD

# Display the provided model
echo "Selected model: $MODEL"
echo "Data path: $DATA_PATH"
echo "Save path: $SAVE_PATH"
echo "Program path: $PROGRAM_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "K fold: $K_FOLD"

## Run the main program
python "${PROGRAM_PATH}" TRAIN -i "${DATA_PATH}" -s "${SAVE_PATH}" -m "${MODEL}" -b "${BATCH_SIZE}" -e "${EPOCHS}" -r "${LEARNING_RATE}" -k "${K_FOLD}"
