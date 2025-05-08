#!/bin/bash
# Setup script for the GraphRAG API Service

# Exit on error
set -e

# Make script executable
chmod +x setup.sh

echo "Setting up GraphRAG API Service..."

# Create required directories
mkdir -p db companies graph prompts data/uploads static temp_mm_chat

# Check if Python virtual environment exists and create if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install or upgrade pip
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create default system configuration file
echo "Creating default system configuration..."
mkdir -p companies
cat > companies/system_config.json << EOL
{
  "default_llm_model": "llama3:8b",
  "default_embedding_model": "mxbai-embed-large:latest",
  "temperature": 0.7,
  "top_k": 4,
  "chunk_size": 1000,
  "chunk_overlap": 200
}
EOL

# Create default company configuration
echo "Creating default company configuration..."
cat > companies/config.json << EOL
{
  "companies": {
    "default": {
      "name": "Default Company",
      "description": "Default company for RAG system",
      "db_dir": "db/default",
      "llm_model": "llama3:8b",
      "embedding_model": "mxbai-embed-large:latest"
    }
  },
  "active_company": "default"
}
EOL

# Create default db directory
mkdir -p db/default

echo "Setup complete!"
echo "To run the API server, execute:"
echo "./run.sh"
