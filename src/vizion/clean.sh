#!/usr/bin/env bash

# Exit immediately if a command fails or a variable is undefined
set -euo pipefail

# Check for input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file.csv> <output_file.csv>" >&2
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

sed -e 's/\r//g' \  # removes carrage return
    -e 's/^[ \t]*//; s/[ \t]*$//' \ # tabs and space normalization
    -e 's/;/,/g' \  #comma substitution
    -e '/^$/d' \ # removing empty lines
    "$INPUT_FILE" > "$OUTPUT_FILE"

# 2. Validation Check: Ensure the clean file actually contains data
if [ ! -s "$OUTPUT_FILE" ]; then
    echo "Validation Error ❗: Cleaned output file is empty!" >&2
    exit 2
fi

echo "Success 🪃 : $INPUT_FILE cleaned and saved to $OUTPUT_FILE"

