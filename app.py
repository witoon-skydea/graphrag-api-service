#!/usr/bin/env python3
"""
GraphRAG API Service - REST API for GraphRAG functionality
"""
import os
import tempfile
import logging
import json
import shutil
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from pydantic import BaseModel, Field

from rag.document_loader import load_document, scan_directory, is_supported_file
from rag.vector_store import (
    get_vector_store, add_documents, similarity_search,
    list_companies, get_active_company, add_company, remove_company, set_active_company,
    set_company_models, get_system_settings, set_system_settings
)
from rag.llm import get_llm_model, generate_response, list_available_llm_models
from rag.embeddings import get_embeddings_model, list_available_embedding_models
from rag.knowledge_graph import KnowledgeGraph
from rag.retrieval import hybrid_retrieval
from rag.visualization import visualize_graph, visualize_query_path
from rag.config import CompanyConfig, SystemConfig

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_GRAPH_DIR = "graph"
API_VERSION = "v1"
UPLOAD_DIR = "data/uploads"
TEMP_DIR = "temp_mm_chat"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="GraphRAG API Service",
    description="REST API for GraphRAG functionality with model selection and knowledge graph capabilities",
    version="1.0.0",
    docs_url=f"/{API_VERSION}/docs",
    redoc_url=f"/{API_VERSION}/redoc",
    openapi_url=f"/{API_VERSION}/openapi.json"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for visualizations
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API requests and responses
class SystemSettings(BaseModel):
    default_llm_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class CompanyCreate(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    set_active: Optional[bool] = False

class CompanyModelUpdate(BaseModel):
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    company_id: Optional[str] = None
    retrieval_method: Optional[str] = "hybrid"
    num_chunks: Optional[int] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    num_hops: Optional[int] = 1
    explain: Optional[bool] = False
    format: Optional[str] = "png"

class IngestOptions(BaseModel):
    company_id: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    build_graph: Optional[bool] = False
    llm_model: Optional[str] = None
    visualize_graph: Optional[bool] = False
    format: Optional[str] = "png"
    recursive: Optional[bool] = True
    ocr: Optional[bool] = False
    ocr_engine: Optional[str] = "tesseract"
    ocr_lang: Optional[str] = "eng"
    ocr_dpi: Optional[int] = 300
    gpu: Optional[bool] = True

class BuildGraphOptions(BaseModel):
    company_id: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    num_docs: Optional[int] = 50
    query: Optional[str] = None
    visualize: Optional[bool] = False
    max_nodes: Optional[int] = 50
    format: Optional[str] = "png"

class VisualizationOptions(BaseModel):
    company_id: Optional[str] = None
    max_nodes: Optional[int] = 50
    format: Optional[str] = "png"

# API routes
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "GraphRAG API Service",
        "version": "1.0.0",
        "endpoints": f"/{API_VERSION}/docs"
    }

@app.get(f"/{API_VERSION}/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "1.0.0"}

# System settings endpoints
@app.get(f"/{API_VERSION}/settings")
async def get_settings():
    """Get system settings"""
    try:
        settings = get_system_settings()
        return {"settings": settings}
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put(f"/{API_VERSION}/settings")
async def update_settings(settings: SystemSettings):
    """Update system settings"""
    try:
        settings_dict = {k: v for k, v in settings.dict().items() if v is not None}
        
        if not settings_dict:
            return {"message": "No settings provided to update"}
            
        set_system_settings(settings_dict)
        updated_settings = get_system_settings()
        
        return {"message": "Settings updated successfully", "settings": updated_settings}
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model endpoints
@app.get(f"/{API_VERSION}/models/llm")
async def list_llm_models():
    """List available LLM models"""
    try:
        models = list_available_llm_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing LLM models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{API_VERSION}/models/embeddings")
async def list_embedding_models():
    """List available embedding models"""
    try:
        models = list_available_embedding_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing embedding models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Company endpoints
@app.get(f"/{API_VERSION}/companies")
async def get_companies():
    """List all companies"""
    try:
        companies = list_companies()
        return {"companies": companies}
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{API_VERSION}/companies/active")
async def get_active_company_endpoint():
    """Get active company"""
    try:
        active = get_active_company()
        return {"active_company": active}
    except Exception as e:
        logger.error(f"Error getting active company: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"/{API_VERSION}/companies")
async def create_company(company: CompanyCreate):
    """Create a new company"""
    try:
        add_company(
            company.id, 
            company.name, 
            company.description, 
            company.llm_model, 
            company.embedding_model
        )
        
        if company.set_active:
            set_active_company(company.id)
            
        # Get updated company list
        companies = list_companies()
        active_company = get_active_company()
        
        return {
            "message": f"Company '{company.name}' created successfully", 
            "company_id": company.id,
            "is_active": company.set_active,
            "active_company": active_company
        }
    except ValueError as e:
        logger.error(f"Error creating company: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating company: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"/{API_VERSION}/companies/{{company_id}}")
async def delete_company(company_id: str):
    """Delete a company"""
    try:
        remove_company(company_id)
        return {"message": f"Company '{company_id}' deleted successfully"}
    except ValueError as e:
        logger.error(f"Error deleting company: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting company: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put(f"/{API_VERSION}/companies/active/{{company_id}}")
async def set_active_company_endpoint(company_id: str):
    """Set active company"""
    try:
        set_active_company(company_id)
        active = get_active_company()
        return {"message": f"Company '{company_id}' set as active", "active_company": active}
    except ValueError as e:
        logger.error(f"Error setting active company: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting active company: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put(f"/{API_VERSION}/companies/{{company_id}}/models")
async def update_company_models(company_id: str, models: CompanyModelUpdate):
    """Update company models"""
    try:
        set_company_models(company_id, models.llm_model, models.embedding_model)
        
        # Get updated model settings
        config = CompanyConfig()
        model_settings = config.get_company_model_settings(company_id)
        
        return {
            "message": f"Models updated for company '{company_id}'", 
            "models": model_settings
        }
    except ValueError as e:
        logger.error(f"Error updating company models: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating company models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint for ingestion
@app.post(f"/{API_VERSION}/ingest")
async def ingest_files(
    files: List[UploadFile] = File(...),
    options: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Ingest documents
    Upload files and ingest them into the vector store and optionally the knowledge graph
    
    Args:
        files: List of files to ingest
        options: JSON string with ingestion options
    """
    try:
        # Parse options
        options_dict = json.loads(options)
        ingest_options = IngestOptions(**options_dict)
        
        # Save files to temporary directory
        saved_files = []
        for file in files:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
            
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
                
            saved_files.append(file_path)
            
        logger.info(f"Saved {len(saved_files)} files for ingestion")
        
        # Process ingestion
        result = process_ingestion(saved_files, ingest_options)
        
        return result
    except Exception as e:
        logger.error(f"Error ingesting files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_ingestion(files: List[str], options: IngestOptions) -> Dict:
    """
    Process document ingestion
    
    Args:
        files: List of file paths to ingest
        options: Ingestion options
        
    Returns:
        Dict with ingestion results
    """
    # Get vector store for the specified company
    config = CompanyConfig()
    
    if options.company_id:
        try:
            db_path = config.get_db_path(options.company_id)
            company_models = config.get_company_model_settings(options.company_id)
            embedding_model = options.embedding_model or company_models.get("embedding_model")
            vector_store = get_vector_store(db_path, embedding_model=embedding_model)
            active_company = options.company_id
        except ValueError as e:
            logger.error(f"Company error: {e}")
            raise ValueError(f"Company error: {e}")
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
        company_models = config.get_company_model_settings()
        embedding_model = options.embedding_model or company_models.get("embedding_model")
        vector_store = get_vector_store(db_path, embedding_model=embedding_model)
    
    # Get knowledge graph if building graph is requested
    knowledge_graph = None
    if options.build_graph:
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        knowledge_graph = KnowledgeGraph(graph_dir)
    
    # Filter out unsupported files
    files_to_ingest = [f for f in files if is_supported_file(f)]
    
    if not files_to_ingest:
        raise ValueError("No supported files to ingest")
    
    # Set up OCR options if OCR is enabled
    ocr_options = None
    if options.ocr:
        ocr_options = {
            'engine': options.ocr_engine,
            'lang': options.ocr_lang,
            'dpi': options.ocr_dpi,
            'use_gpu': options.gpu
        }
    
    # Get chunk settings from system config
    sys_config = SystemConfig()
    chunk_settings = sys_config.get_chunk_settings()
    chunk_size = options.chunk_size or chunk_settings.get("chunk_size")
    chunk_overlap = options.chunk_overlap or chunk_settings.get("chunk_overlap")
    
    # Load and add documents
    total_files = len(files_to_ingest)
    successful_files = 0
    all_documents = []  # Store all documents for knowledge graph building
    
    for i, file_path in enumerate(files_to_ingest, 1):
        logger.info(f"[{i}/{total_files}] Loading {file_path}...")
        try:
            documents = load_document(
                file_path, 
                ocr_enabled=options.ocr, 
                ocr_options=ocr_options,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            add_documents(vector_store, documents)
            all_documents.extend(documents)
            successful_files += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Build knowledge graph if requested
    graph_viz_path = None
    if options.build_graph and all_documents:
        logger.info("Building knowledge graph from documents...")
        llm = get_llm_model(options.llm_model or company_models.get("llm_model"))
        knowledge_graph.extract_and_add_from_documents(all_documents, llm)
        
        # Visualize graph if requested
        if options.visualize_graph:
            # Determine output format
            output_format = options.format.lower()
            if output_format == "mermaid":
                graph_viz_path = os.path.join("static", f"knowledge_graph_{active_company}.md")
            else:
                graph_viz_path = os.path.join("static", f"knowledge_graph_{active_company}.png")
                
            logger.info(f"Visualizing knowledge graph to {graph_viz_path}")
            # Ensure static directory exists
            os.makedirs("static", exist_ok=True)
            visualize_graph(knowledge_graph, graph_viz_path, max_nodes=50, format=output_format)
    
    # Return results
    result = {
        "message": "Ingestion complete",
        "successful_files": successful_files,
        "total_files": total_files,
        "active_company": active_company,
        "vector_store_location": db_path,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "build_graph": options.build_graph
    }
    
    if graph_viz_path:
        result["graph_visualization"] = f"/{graph_viz_path}"
    
    return result

# Query endpoint
@app.post(f"/{API_VERSION}/query")
async def query_endpoint(query: QueryRequest):
    """
    Answer a question using the GraphRAG system
    
    Args:
        query: Query request with question and options
    """
    try:
        # Get vector store and LLM for the specified company
        config = CompanyConfig()
        sys_config = SystemConfig()
        
        if query.company_id:
            try:
                db_path = config.get_db_path(query.company_id)
                company_models = config.get_company_model_settings(query.company_id)
                active_company = query.company_id
            except ValueError as e:
                error_msg = f"Error: {e}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
        else:
            active_company = config.get_active_company()
            db_path = config.get_db_path()
            company_models = config.get_company_model_settings()
        
        if not os.path.exists(db_path):
            error_msg = f"Vector store not found at {db_path}. Please ingest documents first."
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Get embedding model (default or override)
        embedding_model = query.embedding_model or company_models.get("embedding_model")
        
        # Get company details
        company_details = config.get_company_details(active_company)
        
        # Get vector store
        vector_store = get_vector_store(db_path, embedding_model=embedding_model)
        
        # Get top_k from arguments or system settings
        top_k = query.num_chunks or sys_config.get_top_k()
        
        # Get knowledge graph if using graph or hybrid retrieval
        knowledge_graph = None
        graph_data = None
        if query.retrieval_method in ["graph", "hybrid"]:
            graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
            if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
                logger.warning(f"Knowledge graph not found at {graph_dir}. Falling back to vector search.")
                query.retrieval_method = "vector"
            else:
                knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Get documents based on retrieval method
        if query.retrieval_method == "vector" or knowledge_graph is None:
            # Use vector search only
            logger.info(f"Using vector search with embedding model: {embedding_model}")
            documents = similarity_search(vector_store, query.question, k=top_k)
        elif query.retrieval_method == "graph":
            # Use graph search only
            logger.info("Using graph search only")
            # Get LLM for entity extraction
            llm_model = query.llm_model or company_models.get("llm_model")
            temperature = query.temperature or sys_config.get_temperature()
            llm = get_llm_model(llm_model, temperature)
            
            from rag.llm.llm import extract_query_entities
            entities = extract_query_entities(llm, query.question)
            
            graph_data = []
            for entity in entities:
                entity_ids = knowledge_graph.search_entities(entity, limit=2)
                for entity_id in entity_ids:
                    neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=query.num_hops)
                    graph_data.extend(neighbors)
            
            # Convert to documents format
            from rag.retrieval import convert_graph_to_documents
            documents = convert_graph_to_documents(graph_data)
        else:
            # Use hybrid search (default)
            logger.info("Using hybrid search (vector + knowledge graph)")
            # Get LLM for hybrid search
            llm_model = query.llm_model or company_models.get("llm_model")
            temperature = query.temperature or sys_config.get_temperature()
            llm = get_llm_model(llm_model, temperature)
            
            documents = hybrid_retrieval(
                vector_store, 
                knowledge_graph, 
                query.question, 
                llm,
                k=top_k,
                max_hops=query.num_hops
            )
            
            # Extract graph data for explanation if needed
            if query.explain:
                graph_data = []
                for doc in documents:
                    if doc.metadata.get('source') == 'knowledge_graph':
                        entity_id = doc.metadata.get('entity_id')
                        if entity_id:
                            neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                            for item in neighbors:
                                if item not in graph_data:
                                    graph_data.append(item)
        
        # Get LLM model (default or override)
        llm_model = query.llm_model or company_models.get("llm_model")
        temperature = query.temperature or sys_config.get_temperature()
        
        logger.info(f"Using LLM model: {llm_model} (temperature: {temperature})")
        llm = get_llm_model(llm_model, temperature)
        
        # Generate response
        logger.info("Generating response...")
        response = generate_response(llm, documents, query.question, None, graph_data)
        
        # Visualize graph used in query if requested
        explanation_path = None
        if query.explain and graph_data and knowledge_graph:
            # Determine output format
            output_format = query.format.lower()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            if output_format == "mermaid":
                output_path = os.path.join("static", f"query_explanation_{timestamp}.md")
            else:
                output_path = os.path.join("static", f"query_explanation_{timestamp}.png")
            
            # Ensure static directory exists
            os.makedirs("static", exist_ok=True)
            
            logger.info(f"Generating visual explanation to {output_path}")
            visualize_query_path(knowledge_graph, graph_data, output_path, format=output_format)
            explanation_path = f"/{output_path}"
        
        # Create result
        result = {
            "question": query.question,
            "answer": response.strip(),
            "company_id": active_company,
            "company_name": company_details.get("name"),
            "retrieval_method": query.retrieval_method,
            "num_chunks": len(documents),
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "temperature": temperature
        }
        
        if explanation_path:
            result["explanation_visualization"] = explanation_path
        
        # Include document source metadata
        sources = []
        for i, doc in enumerate(documents):
            source = {
                "index": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "chunk": doc.metadata.get("chunk", None)
            }
            
            if doc.metadata.get("source") == "knowledge_graph":
                source["entity_type"] = doc.metadata.get("entity_type", None)
                source["entity_name"] = doc.metadata.get("entity_name", None)
            
            sources.append(source)
        
        result["sources"] = sources
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Build graph endpoint
@app.post(f"/{API_VERSION}/build-graph")
async def build_graph_endpoint(options: BuildGraphOptions):
    """
    Build knowledge graph from existing vector store
    
    Args:
        options: Build graph options
    """
    try:
        # Get vector store and LLM for the specified company
        config = CompanyConfig()
        
        if options.company_id:
            try:
                db_path = config.get_db_path(options.company_id)
                company_models = config.get_company_model_settings(options.company_id)
                active_company = options.company_id
            except ValueError as e:
                error_msg = f"Error: {e}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
        else:
            active_company = config.get_active_company()
            db_path = config.get_db_path()
            company_models = config.get_company_model_settings()
        
        if not os.path.exists(db_path):
            error_msg = f"Vector store not found at {db_path}. Please ingest documents first."
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Get vector store
        embedding_model = options.embedding_model or company_models.get("embedding_model")
        vector_store = get_vector_store(db_path, embedding_model=embedding_model)
        
        # Get knowledge graph
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Get LLM model
        llm_model = options.llm_model or company_models.get("llm_model")
        llm = get_llm_model(llm_model)
        
        # Get documents from vector store
        logger.info("Retrieving documents from vector store")
        
        if options.query:
            logger.info(f"Using query: {options.query}")
            # Use the provided query to find relevant documents
            documents = similarity_search(vector_store, options.query, k=options.num_docs)
        else:
            # Use a generic query to retrieve documents
            logger.info("Using a generic query to retrieve documents")
            documents = similarity_search(vector_store, "summarize all information", k=options.num_docs)
        
        logger.info(f"Retrieved {len(documents)} documents from vector store")
        
        # Build knowledge graph
        logger.info("Building knowledge graph from documents")
        knowledge_graph.extract_and_add_from_documents(documents, llm)
        
        # Visualize graph if requested
        graph_viz_path = None
        if options.visualize:
            # Determine output format
            output_format = options.format.lower()
            if output_format == "mermaid":
                output_path = os.path.join("static", f"knowledge_graph_{active_company}.md")
            else:
                output_path = os.path.join("static", f"knowledge_graph_{active_company}.png")
                
            # Ensure static directory exists
            os.makedirs("static", exist_ok=True)
                
            logger.info(f"Visualizing knowledge graph to {output_path}")
            visualize_graph(knowledge_graph, output_path, max_nodes=options.max_nodes, format=output_format)
            graph_viz_path = f"/{output_path}"
        
        return {
            "message": "Knowledge graph built successfully",
            "company_id": active_company,
            "documents_processed": len(documents),
            "graph_location": graph_dir,
            "visualization": graph_viz_path
        }
    
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Visualize graph endpoint
@app.post(f"/{API_VERSION}/visualize-graph")
async def visualize_graph_endpoint(options: VisualizationOptions):
    """
    Visualize knowledge graph
    
    Args:
        options: Visualization options
    """
    try:
        # Get company path
        config = CompanyConfig()
        
        if options.company_id:
            try:
                db_path = config.get_db_path(options.company_id)
                active_company = options.company_id
            except ValueError as e:
                error_msg = f"Error: {e}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
        else:
            active_company = config.get_active_company()
            db_path = config.get_db_path()
        
        # Get knowledge graph
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
            error_msg = f"Knowledge graph not found at {graph_dir}. Please build the knowledge graph first."
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Determine output format and path
        output_format = options.format.lower()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if output_format == "mermaid":
            output_path = os.path.join("static", f"knowledge_graph_{active_company}_{timestamp}.md")
        else:
            output_path = os.path.join("static", f"knowledge_graph_{active_company}_{timestamp}.png")
        
        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)
        
        logger.info(f"Visualizing knowledge graph to {output_path}")
        
        # Visualize graph
        visualize_graph(knowledge_graph, output_path, max_nodes=options.max_nodes, format=output_format)
        
        return {
            "message": "Knowledge graph visualization created",
            "company_id": active_company,
            "visualization": f"/{output_path}",
            "format": output_format
        }
    
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=6989, reload=True)
