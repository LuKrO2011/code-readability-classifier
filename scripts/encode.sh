#!/bin/bash

# Get the model name from the environment variable
DATA_PATH=$DATA_PATH
ENCODED_PATH=$ENCODED_PATH
SAVE_PATH=$SAVE_PATH

# Display the provided model
echo "Data path: $DATA_PATH"
echo "Encoded path: $ENCODED_PATH"
echo "Save path: $SAVE_PATH"

## Run the main program
python "${PROGRAM_PATH}" ENCODE -i "${DATA_PATH}" -s "${SAVE_PATH}" --intermediate "${ENCODED_PATH}"
