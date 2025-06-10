#!/usr/bin/env python3
"""Test the dimension of CodeBERT fallback model"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_codebert_dimension():
    """Test what dimension CodeBERT produces"""
    print("=" * 80)
    print("TESTING CODEBERT DIMENSION")
    print("=" * 80)
    
    from sentence_transformers import SentenceTransformer
    
    print("\nLoading microsoft/codebert-base...")
    
    try:
        # CodeBERT needs mean pooling
        model = SentenceTransformer('microsoft/codebert-base')
        
        # Test encoding
        test_text = "def hello(): pass"
        embedding = model.encode(test_text)
        
        print(f"CodeBERT dimension: {embedding.shape[-1]}D")
        
        if embedding.shape[-1] == 768:
            print("✓ CodeBERT produces 768D embeddings - compatible with CodeRankEmbed!")
        else:
            print("✗ CodeBERT dimension mismatch!")
            
    except Exception as e:
        print(f"Error loading CodeBERT: {e}")
        print("\nCodeBERT may need special initialization...")

if __name__ == "__main__":
    test_codebert_dimension()