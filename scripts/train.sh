#!/bin/bash

# Get the model name from the environment variable
MODEL=$MODEL
DATA_PATH=$DATA_PATH
SAVE_PATH=$SAVE_PATH
PROGRAM_PATH=$PROGRAM_PATH

# Display the provided model
echo "Selected model: $MODEL"
echo "Data path: $DATA_PATH"
echo "Save path: $SAVE_PATH"
echo "Program path: $PROGRAM_PATH"

## Run the main program
python "${PROGRAM_PATH}" TRAIN -i "${DATA_PATH}" -s "${SAVE_PATH}" -m "${MODEL}"
