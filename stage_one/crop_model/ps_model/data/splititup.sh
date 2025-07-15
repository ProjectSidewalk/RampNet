#!/bin/bash

# Check if the correct number of arguments is given
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_folder>"
    exit 1
fi

SOURCE_FOLDER="$1"
TRAIN_FOLDER="train"
VAL_FOLDER="val"
TEST_FOLDER="test"

# Create train, val, and test directories
mkdir -p "$TRAIN_FOLDER" "$VAL_FOLDER" "$TEST_FOLDER"

# Get a list of all files in the source folder
FILES=($(find "$SOURCE_FOLDER" -type f))
TOTAL_FILES=${#FILES[@]}

# Calculate split sizes
TRAIN_SIZE=$((TOTAL_FILES * 70 / 100))
VAL_SIZE=$((TOTAL_FILES * 15 / 100))
# Remaining goes to test
TEST_SIZE=$((TOTAL_FILES - TRAIN_SIZE - VAL_SIZE))

# Shuffle files
shuffled_files=( $(printf "%s\n" "${FILES[@]}" | shuf) )

# Copy files to train, val, and test folders
for i in "${!shuffled_files[@]}"; do
    FILE_PATH="${shuffled_files[$i]}"
    RELATIVE_PATH="${FILE_PATH#$SOURCE_FOLDER/}"
    
    if [ "$i" -lt "$TRAIN_SIZE" ]; then
        DEST_DIR="$TRAIN_FOLDER/$(dirname "$RELATIVE_PATH")"
    elif [ "$i" -lt $((TRAIN_SIZE + VAL_SIZE)) ]; then
        DEST_DIR="$VAL_FOLDER/$(dirname "$RELATIVE_PATH")"
    else
        DEST_DIR="$TEST_FOLDER/$(dirname "$RELATIVE_PATH")"
    fi

    mkdir -p "$DEST_DIR"
    cp "$FILE_PATH" "$DEST_DIR"
done

echo "Dataset split into train (70%), val (15%), and test (15%) complete!"
