#!/usr/bin/env python3
"""
GraphRAG API Client - Simple client for interacting with the GraphRAG API Service
"""
import os
import sys
import argparse
import json
import requests
from pprint import pprint

# Default API endpoint
API_BASE_URL = "http://localhost:6989/v1"

class GraphRAGClient:
    """Client for the GraphRAG API Service"""
    
    def __init__(self, base_url=API_BASE_URL):
        """Initialize the client"""
        self.base_url = base_url
        
    def get_settings(self):
        """Get system settings"""
        response = requests.get(f"{self.base_url}/settings")
        return response.json()
    
    def update_settings(self, settings):
        """Update system settings"""
        response = requests.put(f"{self.base_url}/settings", json=settings)
        return response.json()
    
    def list_companies(self):
        """List all companies"""
        response = requests.get(f"{self.base_url}/companies")
        return response.json()
    
    def get_active_company(self):
        """Get active company"""
        response = requests.get(f"{self.base_url}/companies/active")
        return response.json()
    
    def create_company(self, company_data):
        """Create a new company"""
        response = requests.post(f"{self.base_url}/companies", json=company_data)
        return response.json()
    
    def delete_company(self, company_id):
        """Delete a company"""
        response = requests.delete(f"{self.base_url}/companies/{company_id}")
        return response.json()
    
    def set_active_company(self, company_id):
        """Set active company"""
        response = requests.put(f"{self.base_url}/companies/active/{company_id}")
        return response.json()
    
    def update_company_models(self, company_id, models):
        """Update company models"""
        response = requests.put(f"{self.base_url}/companies/{company_id}/models", json=models)
        return response.json()
    
    def ingest_files(self, files, options):
        """Ingest documents"""
        file_list = []
        for file_path in files:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_list.append(("files", (os.path.basename(file_path), f, "application/octet-stream")))
            else:
                print(f"File not found: {file_path}")
                
        if not file_list:
            raise ValueError("No valid files to ingest")
            
        response = requests.post(
            f"{self.base_url}/ingest",
            files=file_list,
            data={"options": json.dumps(options)}
        )
        
        return response.json()
    
    def query(self, query_data):
        """Answer a question using the GraphRAG system"""
        response = requests.post(f"{self.base_url}/query", json=query_data)
        return response.json()
    
    def build_graph(self, options):
        """Build knowledge graph from existing vector store"""
        response = requests.post(f"{self.base_url}/build-graph", json=options)
        return response.json()
    
    def visualize_graph(self, options):
        """Visualize knowledge graph"""
        response = requests.post(f"{self.base_url}/visualize-graph", json=options)
        return response.json()

def main():
    """Main function for the CLI client"""
    parser = argparse.ArgumentParser(description="GraphRAG API Client")
    parser.add_argument("--url", default=API_BASE_URL, help="Base URL for API")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Settings commands
    settings_parser = subparsers.add_parser("settings", help="Manage system settings")
    settings_parser.add_argument("--get", action="store_true", help="Get system settings")
    settings_parser.add_argument("--update", help="Update system settings (JSON string)")
    
    # Company commands
    company_parser = subparsers.add_parser("company", help="Manage companies")
    company_parser.add_argument("--list", action="store_true", help="List all companies")
    company_parser.add_argument("--active", action="store_true", help="Get active company")
    company_parser.add_argument("--create", help="Create a new company (JSON string)")
    company_parser.add_argument("--delete", help="Delete a company")
    company_parser.add_argument("--set-active", help="Set active company")
    company_parser.add_argument("--update-models", nargs=2, metavar=("COMPANY_ID", "MODELS_JSON"), 
                               help="Update company models")
    
    # Ingest commands
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("files", nargs="+", help="Files to ingest")
    ingest_parser.add_argument("--options", required=True, help="Ingestion options (JSON string)")
    
    # Query commands
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--question", required=True, help="Question to answer")
    query_parser.add_argument("--company", help="Company ID to query")
    query_parser.add_argument("--retrieval", default="hybrid", 
                             choices=["vector", "graph", "hybrid"],
                             help="Retrieval method to use")
    query_parser.add_argument("--chunks", type=int, help="Number of chunks to return")
    query_parser.add_argument("--explain", action="store_true", help="Generate visual explanation")
    
    # Graph commands
    graph_parser = subparsers.add_parser("graph", help="Manage knowledge graph")
    graph_parser.add_argument("--build", action="store_true", help="Build knowledge graph")
    graph_parser.add_argument("--company", help="Company ID for graph operation")
    graph_parser.add_argument("--visualize", action="store_true", help="Visualize knowledge graph")
    graph_parser.add_argument("--max-nodes", type=int, default=50, help="Maximum nodes for visualization")
    graph_parser.add_argument("--format", choices=["png", "mermaid"], default="png", 
                             help="Visualization format")
    
    args = parser.parse_args()
    
    # Initialize client
    client = GraphRAGClient(args.url)
    
    # Process commands
    if args.command == "settings":
        if args.get:
            result = client.get_settings()
            pprint(result)
        elif args.update:
            settings = json.loads(args.update)
            result = client.update_settings(settings)
            pprint(result)
        else:
            settings_parser.print_help()
            
    elif args.command == "company":
        if args.list:
            result = client.list_companies()
            pprint(result)
        elif args.active:
            result = client.get_active_company()
            pprint(result)
        elif args.create:
            company_data = json.loads(args.create)
            result = client.create_company(company_data)
            pprint(result)
        elif args.delete:
            result = client.delete_company(args.delete)
            pprint(result)
        elif args.set_active:
            result = client.set_active_company(args.set_active)
            pprint(result)
        elif args.update_models:
            company_id = args.update_models[0]
            models = json.loads(args.update_models[1])
            result = client.update_company_models(company_id, models)
            pprint(result)
        else:
            company_parser.print_help()
            
    elif args.command == "ingest":
        options = json.loads(args.options)
        result = client.ingest_files(args.files, options)
        pprint(result)
        
    elif args.command == "query":
        query_data = {
            "question": args.question,
            "retrieval_method": args.retrieval,
            "explain": args.explain
        }
        
        if args.company:
            query_data["company_id"] = args.company
            
        if args.chunks:
            query_data["num_chunks"] = args.chunks
            
        result = client.query(query_data)
        
        # Pretty print the response
        print("\n==== GraphRAG Query Result ====")
        print(f"Question: {result.get('question')}")
        print(f"\nAnswer: {result.get('answer')}")
        print(f"\nCompany: {result.get('company_name')} ({result.get('company_id')})")
        print(f"Models: LLM={result.get('llm_model')}, Embeddings={result.get('embedding_model')}")
        print(f"Retrieved {len(result.get('sources', []))} chunks using {result.get('retrieval_method')} retrieval")
        
        if "explanation_visualization" in result:
            print(f"\nExplanation visualization: {args.url.replace('/v1', '')}{result['explanation_visualization']}")
            
    elif args.command == "graph":
        if args.build:
            options = {
                "visualize": args.visualize,
                "format": args.format,
                "max_nodes": args.max_nodes
            }
            
            if args.company:
                options["company_id"] = args.company
                
            result = client.build_graph(options)
            pprint(result)
            
        elif args.visualize:
            options = {
                "format": args.format,
                "max_nodes": args.max_nodes
            }
            
            if args.company:
                options["company_id"] = args.company
                
            result = client.visualize_graph(options)
            pprint(result)
            
        else:
            graph_parser.print_help()
            
    else:
        parser.print_help()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
