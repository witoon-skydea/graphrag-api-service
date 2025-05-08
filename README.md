# GraphRAG API Service

A RESTful API service for GraphRAG functionality with model selection and knowledge graph capabilities.

## Overview

This service provides API endpoints for the GraphRAG system, a sophisticated Retrieval-Augmented Generation (RAG) system that combines vector search with knowledge graph capabilities. It allows multiple companies/teams to manage their own vector stores and customize model settings.

## Features

- **Multi-tenant Architecture**: Support for multiple companies with isolated document stores
- **Model Selection**: Customize LLM and embedding models per company
- **Knowledge Graph Integration**: Enhanced retrieval using graph traversal methods
- **Document Ingestion**: Process various document formats, including OCR for images/PDFs
- **Hybrid Retrieval**: Combine vector-based similarity search with knowledge graph traversal
- **Visualization**: Generate and serve knowledge graph visualizations
- **RESTful API**: Clean, well-documented API endpoints

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- A local or remote Ollama instance for LLM and embedding models

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd graphrag-api-service
   ```

2. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

This will:
- Create a Python virtual environment
- Install all dependencies
- Set up default configurations
- Create necessary directories

## Running the Service

Start the API server:

```
chmod +x run.sh
./run.sh
```

This will start the service on port 6989. You can access the API documentation at:
- http://localhost:6989/v1/docs

## API Endpoints

The API service exposes several endpoints:

### System Settings
- `GET /v1/settings` - Get system settings
- `PUT /v1/settings` - Update system settings

### Model Management
- `GET /v1/models/llm` - List available LLM models
- `GET /v1/models/embeddings` - List available embedding models

### Company Management
- `GET /v1/companies` - List all companies
- `GET /v1/companies/active` - Get active company
- `POST /v1/companies` - Create a new company
- `DELETE /v1/companies/{company_id}` - Delete a company
- `PUT /v1/companies/active/{company_id}` - Set active company
- `PUT /v1/companies/{company_id}/models` - Update company models

### Document Management
- `POST /v1/ingest` - Ingest documents

### RAG Operations
- `POST /v1/query` - Answer a question using the GraphRAG system
- `POST /v1/build-graph` - Build knowledge graph from existing vector store
- `POST /v1/visualize-graph` - Visualize knowledge graph

## Directory Structure

- `app.py` - Main API server file
- `rag/` - Core RAG modules
- `companies/` - Company configuration files
- `db/` - Vector stores for each company
- `graph/` - Knowledge graph data
- `prompts/` - Prompt templates
- `data/uploads/` - Temporary directory for uploaded files
- `static/` - Static files for visualizations
- `temp_mm_chat/` - Temporary chat files

## Example Usage

### Ingest Documents

```python
import requests
import json

# API Endpoint
url = "http://localhost:6989/v1/ingest"

# Options for ingestion
options = {
    "company_id": "default",
    "build_graph": True,
    "visualize_graph": True,
    "ocr": True
}

# Files to upload
files = [
    ('files', ('document1.pdf', open('document1.pdf', 'rb'), 'application/pdf')),
    ('files', ('document2.docx', open('document2.docx', 'rb'), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'))
]

# Send request
response = requests.post(
    url,
    files=files,
    data={'options': json.dumps(options)}
)

# Print response
print(response.json())
```

### Query

```python
import requests
import json

# API Endpoint
url = "http://localhost:6989/v1/query"

# Query request
query = {
    "question": "What is the main topic of the documents?",
    "company_id": "default",
    "retrieval_method": "hybrid",
    "explain": True
}

# Send request
response = requests.post(
    url,
    json=query
)

# Print response
print(response.json())
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
