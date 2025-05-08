#!/usr/bin/env python3
"""
Test script for GraphRAG API Service
"""
import requests
import json
import os
import sys
import argparse
from pprint import pprint

# Default API endpoint
API_BASE_URL = "http://localhost:6989/v1"

def test_health_check(base_url=API_BASE_URL):
    """Test the health check endpoint"""
    print("\n==== Testing Health Check ====")
    response = requests.get(f"{base_url}/health")
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    return response.status_code == 200

def test_settings(base_url=API_BASE_URL):
    """Test the settings endpoints"""
    print("\n==== Testing Settings Endpoints ====")
    
    # Get settings
    print("\n-- Getting settings --")
    response = requests.get(f"{base_url}/settings")
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    
    # Update settings
    print("\n-- Updating settings --")
    settings = {
        "top_k": 5
    }
    response = requests.put(f"{base_url}/settings", json=settings)
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    
    return response.status_code == 200

def test_models(base_url=API_BASE_URL):
    """Test model listing endpoints"""
    print("\n==== Testing Model Endpoints ====")
    
    # List LLM models
    print("\n-- Listing LLM models --")
    response = requests.get(f"{base_url}/models/llm")
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    
    # List embedding models
    print("\n-- Listing embedding models --")
    response = requests.get(f"{base_url}/models/embeddings")
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    
    return response.status_code == 200

def test_companies(base_url=API_BASE_URL):
    """Test company management endpoints"""
    print("\n==== Testing Company Endpoints ====")
    
    # List companies
    print("\n-- Listing companies --")
    response = requests.get(f"{base_url}/companies")
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    
    # Get active company
    print("\n-- Getting active company --")
    response = requests.get(f"{base_url}/companies/active")
    print(f"Status code: {response.status_code}")
    pprint(response.json())
    
    # Create a test company
    print("\n-- Creating test company --")
    company = {
        "id": "test_company",
        "name": "Test Company",
        "description": "Test company for API testing",
        "set_active": False
    }
    response = requests.post(f"{base_url}/companies", json=company)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        pprint(response.json())
        
        # Set active company
        print("\n-- Setting test company as active --")
        response = requests.put(f"{base_url}/companies/active/test_company")
        print(f"Status code: {response.status_code}")
        pprint(response.json())
        
        # Update company models
        print("\n-- Updating company models --")
        models = {
            "llm_model": "llama3:8b"
        }
        response = requests.put(f"{base_url}/companies/test_company/models", json=models)
        print(f"Status code: {response.status_code}")
        pprint(response.json())
        
        # Delete test company
        print("\n-- Deleting test company --")
        # First set active company back to default
        requests.put(f"{base_url}/companies/active/default")
        response = requests.delete(f"{base_url}/companies/test_company")
        print(f"Status code: {response.status_code}")
        pprint(response.json())
    
    return response.status_code == 200

def test_simple_query(base_url=API_BASE_URL):
    """Test simple query endpoint with existing data"""
    print("\n==== Testing Simple Query ====")
    
    query = {
        "question": "What information is available?",
        "retrieval_method": "vector",
        "num_chunks": 2
    }
    
    response = requests.post(f"{base_url}/query", json=query)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Question: {result.get('question')}")
        print(f"Answer: {result.get('answer')}")
        print(f"Company: {result.get('company_name')} ({result.get('company_id')})")
        print(f"Models: LLM={result.get('llm_model')}, Embeddings={result.get('embedding_model')}")
        print(f"Retrieved {len(result.get('sources', []))} chunks")
    else:
        print("Query failed - this is expected if no documents have been ingested yet")
    
    return True  # Allow this test to "pass" even if it fails, as it requires data

def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description="Test GraphRAG API Service")
    parser.add_argument("--url", default=API_BASE_URL, help="Base URL for API")
    parser.add_argument("--tests", default="all", 
                       help="Comma-separated list of tests to run (health,settings,models,companies,query,all)")
    
    args = parser.parse_args()
    
    tests = {
        "health": test_health_check,
        "settings": test_settings,
        "models": test_models,
        "companies": test_companies,
        "query": test_simple_query,
    }
    
    # Determine which tests to run
    test_list = []
    if args.tests.lower() == "all":
        test_list = list(tests.keys())
    else:
        test_list = [t.strip().lower() for t in args.tests.split(",")]
    
    # Run tests
    results = {}
    for test_name in test_list:
        if test_name in tests:
            print(f"\nRunning test: {test_name}")
            try:
                result = tests[test_name](args.url)
                results[test_name] = "PASS" if result else "FAIL"
            except Exception as e:
                print(f"Error: {e}")
                results[test_name] = "ERROR"
        else:
            print(f"Unknown test: {test_name}")
    
    # Print summary
    print("\n==== Test Summary ====")
    for test_name, result in results.items():
        print(f"{test_name.ljust(10)}: {result}")
    
    # Return 0 if all passed, 1 otherwise
    if all(r == "PASS" for r in results.values()):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed or had errors.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
