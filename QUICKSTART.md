# GraphRAG API Service: Quick Start Guide

This guide will help you get the GraphRAG API Service up and running quickly.

## Prerequisites

- Python 3.8+ (for local installation)
- Docker and Docker Compose (for Docker installation)
- Ollama (local or remote instance)

## Installation Options

### Option 1: Local Installation

1. **Run the setup script:**

```bash
chmod +x setup.sh
./setup.sh
```

2. **Start the API server:**

```bash
chmod +x run.sh
./run.sh
```

The API will be available at http://localhost:6989/v1/docs

### Option 2: Docker Installation

1. **Build and start the containers:**

```bash
chmod +x docker-run.sh
./docker-run.sh --build --start
```

The API will be available at http://localhost:6989/v1/docs

## Basic Usage

### 1. Verify the API is working

```bash
curl http://localhost:6989/v1/health
```

You should see a response like:
```json
{"status": "ok", "version": "1.0.0"}
```

### 2. Set up a company (or use the default)

The system comes with a default company already configured. You can create a new one:

```bash
python client.py company --create '{"id": "mycompany", "name": "My Company", "description": "My custom RAG system", "set_active": true}'
```

### 3. Ingest documents

```bash
python client.py ingest path/to/document1.pdf path/to/document2.docx --options '{"build_graph": true, "visualize_graph": true}'
```

### 4. Query the system

```bash
python client.py query --question "What does the document say about AI?" --retrieval hybrid --explain
```

## Using the Web API

The API provides a Swagger UI at http://localhost:6989/v1/docs where you can:

- Test all API endpoints
- See request/response schemas
- Execute operations directly from the browser

## Next Steps

- Explore the full README.md for detailed documentation
- Check out the client.py script for more usage examples
- Use test_api.py to verify functionality: `python test_api.py`

## Common Tasks

### Update system settings

```bash
python client.py settings --update '{"temperature": 0.5, "top_k": 5}'
```

### List available models

```bash
# List LLM models
curl http://localhost:6989/v1/models/llm

# List embedding models
curl http://localhost:6989/v1/models/embeddings
```

### Build knowledge graph

```bash
python client.py graph --build --visualize --company mycompany
```

### Visualize existing knowledge graph

```bash
python client.py graph --visualize --format png --max-nodes 50
```

## Troubleshooting

- **API not starting**: Check if port 6989 is already in use
- **Ingestion failing**: Ensure Ollama is running and accessible
- **Docker issues**: Try running `./docker-run.sh --restart` to restart the services
