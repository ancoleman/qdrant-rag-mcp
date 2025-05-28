#!/usr/bin/env python3
"""
Test multi-language AST chunking functionality
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.ast_chunker import create_ast_chunker
from indexers.code_indexer import CodeIndexer

# Test Shell script content
SHELL_SCRIPT = '''#!/bin/bash
# Setup script for testing

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/test.log"

# Utility function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Main setup function
setup_environment() {
    log_message "Starting setup..."
    
    # Create directories
    mkdir -p "$HOME/.config/myapp"
    
    # Set permissions
    chmod 755 "$HOME/.config/myapp"
    
    log_message "Setup complete"
}

# Cleanup function
cleanup() {
    rm -f "$LOG_FILE"
    echo "Cleanup done"
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_environment
    cleanup
fi
'''

# Test Go code content  
GO_CODE = '''package main

import (
    "fmt"
    "log"
    "os"
)

// Config represents application configuration
type Config struct {
    Host     string
    Port     int
    Debug    bool
}

// Logger interface for different logging implementations
type Logger interface {
    Info(msg string)
    Error(msg string)
    Debug(msg string)
}

// SimpleLogger implements the Logger interface
type SimpleLogger struct {
    prefix string
}

// NewSimpleLogger creates a new SimpleLogger instance
func NewSimpleLogger(prefix string) *SimpleLogger {
    return &SimpleLogger{prefix: prefix}
}

// Info logs an info message
func (l *SimpleLogger) Info(msg string) {
    log.Printf("[%s INFO] %s", l.prefix, msg)
}

// Error logs an error message
func (l *SimpleLogger) Error(msg string) {
    log.Printf("[%s ERROR] %s", l.prefix, msg)
}

// Debug logs a debug message
func (l *SimpleLogger) Debug(msg string) {
    if os.Getenv("DEBUG") != "" {
        log.Printf("[%s DEBUG] %s", l.prefix, msg)
    }
}

// LoadConfig loads configuration from environment
func LoadConfig() *Config {
    return &Config{
        Host:  getEnv("HOST", "localhost"),
        Port:  8080,
        Debug: os.Getenv("DEBUG") != "",
    }
}

// getEnv gets environment variable with default
func getEnv(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}

func main() {
    logger := NewSimpleLogger("APP")
    config := LoadConfig()
    
    logger.Info(fmt.Sprintf("Starting server on %s:%d", config.Host, config.Port))
    
    if config.Debug {
        logger.Debug("Debug mode enabled")
    }
}
'''

# Test JavaScript content
JAVASCRIPT_CODE = '''import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import './UserProfile.css';

// Constants
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';
const DEFAULT_AVATAR = '/images/default-avatar.png';

export interface UserData {
    id: string;
    name: string;
    email: string;
    avatar?: string;
    bio?: string;
}

interface UserProfileProps {
    userId: string;
    onUpdate?: (user: UserData) => void;
}

export const UserProfile: React.FC<UserProfileProps> = ({ userId, onUpdate }) => {
    const [user, setUser] = useState<UserData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch user data
    useEffect(() => {
        fetchUser();
    }, [userId]);

    const fetchUser = async () => {
        try {
            setLoading(true);
            const response = await axios.get(`${API_BASE_URL}/users/${userId}`);
            setUser(response.data);
        } catch (err) {
            setError(err.message || 'Failed to load user');
        } finally {
            setLoading(false);
        }
    };

    const updateProfile = async (updates: Partial<UserData>) => {
        try {
            const response = await axios.patch(
                `${API_BASE_URL}/users/${userId}`,
                updates
            );
            setUser(response.data);
            onUpdate?.(response.data);
        } catch (err) {
            setError(err.message || 'Failed to update profile');
        }
    };

    if (loading) return <div className="loading">Loading...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!user) return <div className="not-found">User not found</div>;

    return (
        <div className="user-profile">
            <img 
                src={user.avatar || DEFAULT_AVATAR} 
                alt={user.name}
                className="avatar"
            />
            <h1>{user.name}</h1>
            <p>{user.email}</p>
            {user.bio && <p className="bio">{user.bio}</p>}
        </div>
    );
};

// Helper functions
export function formatUserName(user: UserData): string {
    return user.name.trim() || 'Anonymous';
}

export const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

class UserService {
    private static instance: UserService;
    private cache: Map<string, UserData> = new Map();

    private constructor() {}

    static getInstance(): UserService {
        if (!UserService.instance) {
            UserService.instance = new UserService();
        }
        return UserService.instance;
    }

    async getUser(id: string): Promise<UserData> {
        if (this.cache.has(id)) {
            return this.cache.get(id)!;
        }

        const response = await fetch(`${API_BASE_URL}/users/${id}`);
        const user = await response.json();
        this.cache.set(id, user);
        return user;
    }

    clearCache(): void {
        this.cache.clear();
    }
}

type UserRole = 'admin' | 'user' | 'guest';

interface Permission {
    resource: string;
    actions: string[];
}

type UserPermissions = Record<UserRole, Permission[]>;

export { UserService, type UserRole, type Permission };
'''

def test_shell_chunking():
    """Test Shell script AST chunking"""
    print("Testing Shell Script Chunking...")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(SHELL_SCRIPT)
        temp_path = f.name
    
    try:
        # Test with AST chunker directly
        chunker = create_ast_chunker('.sh')
        chunks = chunker.chunk_file(temp_path)
        
        print(f"\nShell chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"  - {chunk.chunk_type}: {chunk.name} (lines {chunk.line_start}-{chunk.line_end})")
            print(f"    Hierarchy: {' > '.join(chunk.hierarchy)}")
            
        # Test with CodeIndexer
        indexer = CodeIndexer(use_ast_chunking=True)
        code_chunks = indexer.index_file(temp_path)
        
        print(f"\nCodeIndexer shell chunks: {len(code_chunks)}")
        for chunk in code_chunks[:3]:
            meta = chunk.metadata
            print(f"  - {meta.get('chunk_type')}: {meta.get('name')} [{meta.get('language')}]")
            
    finally:
        os.unlink(temp_path)

def test_go_chunking():
    """Test Go code AST chunking"""
    print("\n\nTesting Go Code Chunking...")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
        f.write(GO_CODE)
        temp_path = f.name
    
    try:
        # Test with AST chunker directly
        chunker = create_ast_chunker('.go')
        chunks = chunker.chunk_file(temp_path)
        
        print(f"\nGo chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"  - {chunk.chunk_type}: {chunk.name} (lines {chunk.line_start}-{chunk.line_end})")
            print(f"    Hierarchy: {' > '.join(chunk.hierarchy)}")
            if chunk.metadata.get('is_exported') is not None:
                print(f"    Exported: {chunk.metadata['is_exported']}")
            
        # Test with CodeIndexer
        indexer = CodeIndexer(use_ast_chunking=True)
        code_chunks = indexer.index_file(temp_path)
        
        print(f"\nCodeIndexer Go chunks: {len(code_chunks)}")
        for chunk in code_chunks[:5]:
            meta = chunk.metadata
            print(f"  - {meta.get('chunk_type')}: {meta.get('name')} [{meta.get('language')}]")
            if 'receiver_type' in meta and meta['receiver_type']:
                print(f"    Method of: {meta['receiver_type']}")
                
    finally:
        os.unlink(temp_path)

def test_existing_scripts():
    """Test on existing shell scripts in the project"""
    print("\n\nTesting on Existing Scripts...")
    
    # Test on actual shell scripts
    script_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    test_files = ['setup.sh', 'create-release.sh']
    
    indexer = CodeIndexer(use_ast_chunking=True)
    
    for script_name in test_files:
        script_path = os.path.join(script_dir, script_name)
        if os.path.exists(script_path):
            print(f"\n{script_name}:")
            chunks = indexer.index_file(script_path)
            
            # Group by chunk type
            by_type = {}
            for chunk in chunks:
                ctype = chunk.metadata.get('chunk_type', 'unknown')
                by_type[ctype] = by_type.get(ctype, 0) + 1
            
            print(f"  Total chunks: {len(chunks)}")
            for ctype, count in by_type.items():
                print(f"  - {ctype}: {count}")

def test_javascript_chunking():
    """Test JavaScript/TypeScript AST chunking"""
    print("\n\nTesting JavaScript/TypeScript Chunking...")
    
    # Create temporary file with .tsx extension (TypeScript + JSX)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsx', delete=False) as f:
        f.write(JAVASCRIPT_CODE)
        temp_path = f.name
    
    try:
        # Test with AST chunker directly
        chunker = create_ast_chunker('.tsx')
        chunks = chunker.chunk_file(temp_path)
        
        print(f"\nJS/TS chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"  - {chunk.chunk_type}: {chunk.name} (lines {chunk.line_start}-{chunk.line_end})")
            print(f"    Hierarchy: {' > '.join(chunk.hierarchy)}")
            if 'is_exported' in chunk.metadata:
                print(f"    Exported: {chunk.metadata['is_exported']}")
            if 'is_async' in chunk.metadata:
                print(f"    Async: {chunk.metadata['is_async']}")
            
        # Test with CodeIndexer
        indexer = CodeIndexer(use_ast_chunking=True)
        code_chunks = indexer.index_file(temp_path)
        
        print(f"\nCodeIndexer JS/TS chunks: {len(code_chunks)}")
        # Group by chunk type
        by_type = {}
        for chunk in code_chunks:
            ctype = chunk.metadata.get('chunk_type', 'unknown')
            by_type[ctype] = by_type.get(ctype, 0) + 1
        
        print("Chunk types:")
        for ctype, count in by_type.items():
            print(f"  - {ctype}: {count}")
            
    finally:
        os.unlink(temp_path)

def test_javascript_variants():
    """Test different JavaScript file extensions"""
    print("\n\nTesting JavaScript File Variants...")
    
    simple_js = '''
// Simple JavaScript file
const utils = require('./utils');

function processData(data) {
    if (!data || !Array.isArray(data)) {
        throw new Error('Invalid data');
    }
    
    return data.map(item => ({
        ...item,
        processed: true,
        timestamp: Date.now()
    }));
}

const handler = async (req, res) => {
    try {
        const result = await fetchData(req.params.id);
        res.json(processData(result));
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};

module.exports = { processData, handler };
'''
    
    extensions = ['.js', '.jsx', '.ts']
    
    for ext in extensions:
        print(f"\nTesting {ext} file:")
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
            f.write(simple_js)
            temp_path = f.name
        
        try:
            indexer = CodeIndexer(use_ast_chunking=True)
            chunks = indexer.index_file(temp_path)
            print(f"  Created {len(chunks)} chunks")
            for chunk in chunks[:3]:
                print(f"    - {chunk.metadata.get('chunk_type')}: {chunk.metadata.get('name', 'unnamed')}")
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    test_shell_chunking()
    test_go_chunking()
    test_javascript_chunking()
    test_javascript_variants()
    test_existing_scripts()