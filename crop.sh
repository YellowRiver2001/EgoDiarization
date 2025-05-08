#!/bin/bash

# Define the path
LIGHT_ASD_PATH="Light-ASD"

# Define the Python script name
PYTHON_SCRIPT="Columbia_test.py"

# Define the video name and folder
VIDEO_NAME="0001"
VIDEO_FOLDER="demo"

# Construct the full Python command
PYTHON_COMMAND="python ${LIGHT_ASD_PATH}/${PYTHON_SCRIPT} --videoName ${VIDEO_NAME} --videoFolder ${VIDEO_FOLDER}"

# Execute the Python command
echo "Executing command: ${PYTHON_COMMAND}"
eval "${PYTHON_COMMAND}"

echo "Command execution finished."

exit 0