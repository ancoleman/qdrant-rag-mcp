#!/usr/bin/env python3
"""
HTTP REST API wrapper for the Qdrant MCP RAG Server
This allows testing the RAG server functionality via HTTP requests
"""

import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn

# Import our MCP server
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qdrant_mcp_context_aware import QdrantRAGServer, Request

app = FastAPI(title="Qdrant RAG Server HTTP API", version="1.0.0")

# Global server instance
rag_server = None

class IndexCodeRequest(BaseModel):
    file_path: str

class IndexConfigRequest(BaseModel):
    file_path: str

class IndexDirectoryRequest(BaseModel):
    directory: str
    patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    collections: Optional[List[str]] = None

class SearchCodeRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    language: Optional[str] = None
    chunk_type: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class SearchConfigRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    file_type: Optional[str] = None
    path: Optional[str] = None
    depth: Optional[int] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG server on startup"""
    global rag_server
    print("Initializing RAG server...")
    rag_server = QdrantRAGServer()
    print("RAG server initialized!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Qdrant RAG Server HTTP API", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "server": "qdrant-rag-http-api"}

@app.post("/index_code")
async def index_code(request: IndexCodeRequest):
    """Index a code file"""
    try:
        mcp_request = Request(params={"file_path": request.file_path})
        response = await rag_server.index_code(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index_config")
async def index_config(request: IndexConfigRequest):
    """Index a configuration file"""
    try:
        mcp_request = Request(params={"file_path": request.file_path})
        response = await rag_server.index_config(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index_directory")
async def index_directory(request: IndexDirectoryRequest):
    """Index an entire directory"""
    try:
        params = {"directory": request.directory}
        if request.patterns:
            params["patterns"] = request.patterns
        if request.exclude_patterns:
            params["exclude_patterns"] = request.exclude_patterns
            
        mcp_request = Request(params=params)
        response = await rag_server.index_directory(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    """General search across all collections"""
    try:
        params = {"query": request.query, "n_results": request.n_results}
        if request.collections:
            params["collections"] = request.collections
            
        mcp_request = Request(params=params)
        response = await rag_server.search(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_code")
async def search_code(request: SearchCodeRequest):
    """Search in code collection with filtering"""
    try:
        params = {"query": request.query, "n_results": request.n_results}
        if request.language:
            params["language"] = request.language
        if request.chunk_type:
            params["chunk_type"] = request.chunk_type
        if request.filters:
            params["filters"] = request.filters
            
        mcp_request = Request(params=params)
        response = await rag_server.search_code(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_config")
async def search_config(request: SearchConfigRequest):
    """Search in config collection with filtering"""
    try:
        params = {"query": request.query, "n_results": request.n_results}
        if request.file_type:
            params["file_type"] = request.file_type
        if request.path:
            params["path"] = request.path
        if request.depth:
            params["depth"] = request.depth
            
        mcp_request = Request(params=params)
        response = await rag_server.search_config(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def get_collections():
    """Get information about Qdrant collections"""
    try:
        # Get collection info from Qdrant (synchronous call)
        collections = rag_server.qdrant_client.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)