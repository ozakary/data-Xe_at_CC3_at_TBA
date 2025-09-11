#!/bin/bash

# Check if the directory is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Navigate to the specified directory
cd "$1" || { echo "Directory not found: $1"; exit 1; }

# List all folders starting with 'cluster_', extract the numbers, and sort them
ls -d cluster_* | sed 's/cluster_//' | sort -n
