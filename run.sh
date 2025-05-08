#!/bin/bash
# Run script for the GraphRAG API Service

# Exit on error
set -e

# Make script executable
chmod +x run.sh

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Python virtual environment not found. Running setup first..."
    ./setup.sh
fi

# Activate virtual environment
source venv/bin/activate

# Create static directory if it doesn't exist
mkdir -p static

# Run the API server
echo "Starting GraphRAG API Service on port 6989..."
uvicorn app:app --host 0.0.0.0 --port 6989 --reload
