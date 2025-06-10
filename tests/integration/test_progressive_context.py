"""Tests for Progressive Context Management."""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.progressive_context import (
    QueryIntentClassifier,
    SemanticCache,
    HierarchyBuilder,
    CodeHierarchy,
    QueryIntent,
    ExpansionOption
)


class TestQueryIntentClassifier(unittest.TestCase):
    """Test query intent classification."""
    
    def setUp(self):
        self.classifier = QueryIntentClassifier()
    
    def test_file_level_classification(self):
        """Test queries that should classify as file level."""
        queries = [
            "What does the authentication system do?",
            "Explain the payment module",
            "Give me an overview of the user service",
            "How does the caching system work?",
            "Describe the architecture of the API"
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            self.assertEqual(intent.level, "file", f"Query '{query}' should classify as file level")
            self.assertEqual(intent.exploration_type, "understanding")
    
    def test_method_level_classification(self):
        """Test queries that should classify as method level."""
        queries = [
            "Show me the bug in the login function",
            "There's an error in line 45",
            "Fix the implementation of save_user",
            "Debug the payment processing",
            "The specific issue is in the validation"
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            self.assertEqual(intent.level, "method", f"Query '{query}' should classify as method level")
            self.assertEqual(intent.exploration_type, "debugging")
    
    def test_class_level_classification(self):
        """Test queries that should classify as class level."""
        queries = [
            "Find the UserManager class",
            "Where is the payment handler?",
            "Show me the authentication service",
            "I'm looking for the cache implementation",
            "Locate the database connection class"
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            self.assertEqual(intent.level, "class", f"Query '{query}' should classify as class level")
            self.assertEqual(intent.exploration_type, "navigation")
    
    def test_confidence_scores(self):
        """Test that confidence scores are reasonable."""
        # High confidence query
        intent = self.classifier.classify("What does the authentication system do?")
        self.assertGreaterEqual(intent.confidence, 0.7)
        
        # Lower confidence query (no clear keywords)
        intent = self.classifier.classify("Something about users")
        self.assertLessEqual(intent.confidence, 0.7)


class TestCodeHierarchy(unittest.TestCase):
    """Test code hierarchy building."""
    
    def test_add_file(self):
        """Test adding files to hierarchy."""
        hierarchy = CodeHierarchy()
        hierarchy.add_file("auth/login.py", "Handles user authentication")
        
        self.assertIn("auth/login.py", hierarchy.files)
        self.assertEqual(hierarchy.files["auth/login.py"]["summary"], "Handles user authentication")
    
    def test_add_class(self):
        """Test adding classes to hierarchy."""
        hierarchy = CodeHierarchy()
        hierarchy.add_class("auth/login.py", "LoginManager", "Manages login flow", ["login", "logout"])
        
        self.assertIn("auth/login.py", hierarchy.files)
        self.assertIn("LoginManager", hierarchy.files["auth/login.py"]["classes"])
        self.assertEqual(
            hierarchy.files["auth/login.py"]["classes"]["LoginManager"]["methods"],
            ["login", "logout"]
        )
    
    def test_add_method(self):
        """Test adding methods to hierarchy."""
        hierarchy = CodeHierarchy()
        
        # Add standalone function
        hierarchy.add_method("utils.py", None, "format_date", "def format_date(date: str) -> str:", "Formats date strings")
        self.assertEqual(len(hierarchy.files["utils.py"]["functions"]), 1)
        
        # Add class method
        hierarchy.add_class("models.py", "User", "User model")
        hierarchy.add_method("models.py", "User", "save", "def save(self) -> bool:", "Saves user to database")
        self.assertIn("save", hierarchy.files["models.py"]["classes"]["User"]["methods"])


class TestHierarchyBuilder(unittest.TestCase):
    """Test hierarchy builder functionality."""
    
    def setUp(self):
        self.builder = HierarchyBuilder()
    
    def test_build_from_results(self):
        """Test building hierarchy from search results."""
        search_results = [
            {
                "file_path": "auth/login.py",
                "chunk_type": "class",
                "metadata": {
                    "name": "LoginManager",
                    "methods": ["login", "logout", "validate"]
                },
                "content": "class LoginManager:\n    '''Manages user login'''",
            },
            {
                "file_path": "auth/login.py",
                "chunk_type": "function",
                "metadata": {
                    "name": "login",
                    "parent_class": "LoginManager",
                    "signature": "def login(self, username: str, password: str) -> bool:"
                },
                "content": "def login(self, username: str, password: str) -> bool:\n    '''Authenticate user'''",
            }
        ]
        
        hierarchy = self.builder.build(search_results)
        
        self.assertIn("auth/login.py", hierarchy.files)
        self.assertIn("LoginManager", hierarchy.files["auth/login.py"]["classes"])
        self.assertIn("login", hierarchy.files["auth/login.py"]["classes"]["LoginManager"]["methods"])
    
    def test_extract_summary(self):
        """Test summary extraction from results."""
        # Test with docstring
        result = {
            "metadata": {"docstring": "This is a test function.\nIt does testing."},
            "content": "def test():\n    pass"
        }
        summary = self.builder._extract_summary(result)
        self.assertEqual(summary, "This is a test function.")
        
        # Test without docstring but with content
        result = {
            "metadata": {},
            "content": 'def test():\n    """Test function for testing."""\n    pass'
        }
        summary = self.builder._extract_summary(result)
        self.assertEqual(summary, "Test function for testing.")


if __name__ == "__main__":
    unittest.main()