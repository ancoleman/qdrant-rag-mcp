"""
Code Generator for GitHub Issue Resolution

Generates code fixes and suggestions based on issue analysis and RAG search results.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Generates code fixes and suggestions for GitHub issues.
    
    Uses analysis results and codebase patterns to suggest appropriate fixes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the code generator.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.safety_config = self.config.get("safety", {})
        
        # Safety limits
        self.max_files_per_pr = self.safety_config.get("max_files_per_pr", 10)
        self.max_lines_per_file = self.safety_config.get("max_lines_per_file", 1000)
        self.blocked_patterns = self.safety_config.get("blocked_file_patterns", [])
        
        # Common code templates
        self.templates = {
            "error_handling": {
                "python": """try:
    {original_code}
except {exception_type} as e:
    logger.error(f"Error in {function_name}: {{e}}")
    {error_action}""",
                "javascript": """try {
    {original_code}
} catch (error) {
    console.error(`Error in {function_name}:`, error);
    {error_action}
}""",
                "java": """try {
    {original_code}
} catch ({exception_type} e) {
    logger.error("Error in {function_name}: " + e.getMessage());
    {error_action}
}"""
            },
            "null_check": {
                "python": """if {variable} is None:
    {null_action}
    return{return_value}""",
                "javascript": """if ({variable} === null || {variable} === undefined) {
    {null_action}
    return{return_value};
}""",
                "java": """if ({variable} == null) {
    {null_action}
    return{return_value};
}"""
            },
            "validation": {
                "python": """if not {condition}:
    raise ValueError("{error_message}")""",
                "javascript": """if (!({condition})) {
    throw new Error("{error_message}");
}""",
                "java": """if (!({condition})) {
    throw new IllegalArgumentException("{error_message}");
}"""
            }
        }
    
    def generate_fix_suggestions(self, issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code fix suggestions based on issue analysis.
        
        Args:
            issue_analysis: Results from IssueAnalyzer
            
        Returns:
            Generated fix suggestions and code changes
        """
        try:
            suggestions = {
                "fixes": [],
                "improvements": [],
                "new_files": [],
                "file_modifications": [],
                "test_suggestions": [],
                "safety_warnings": [],
                "confidence_level": "medium",
                "generated_at": datetime.now().isoformat()
            }
            
            issue_type = issue_analysis["extracted_info"]["issue_type"]
            relevant_files = issue_analysis["analysis"]["relevant_files"]
            
            # Generate fixes based on issue type
            if issue_type == "bug":
                suggestions.update(self._generate_bug_fixes(issue_analysis))
            elif issue_type == "feature":
                suggestions.update(self._generate_feature_implementation(issue_analysis))
            elif issue_type == "performance":
                suggestions.update(self._generate_performance_improvements(issue_analysis))
            
            # Add general improvements
            suggestions["improvements"].extend(self._generate_general_improvements(relevant_files))
            
            # Generate test suggestions
            suggestions["test_suggestions"] = self._generate_test_suggestions(issue_analysis)
            
            # Perform safety checks
            suggestions["safety_warnings"] = self._perform_safety_checks(suggestions)
            
            # Calculate confidence level
            suggestions["confidence_level"] = self._calculate_confidence(issue_analysis)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate fix suggestions: {e}")
            raise
    
    def _generate_bug_fixes(self, issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific fixes for bug reports."""
        fixes = []
        file_modifications = []
        
        errors = issue_analysis["extracted_info"].get("errors", [])
        relevant_files = issue_analysis["analysis"]["relevant_files"]
        
        # Analyze error patterns
        for error in errors[:3]:  # Limit to top 3 errors
            error_fixes = self._analyze_error_pattern(error)
            fixes.extend(error_fixes)
        
        # Generate file modifications for common bug patterns
        for file_info in relevant_files[:5]:
            file_path = file_info["file_path"]
            language = self._detect_language(file_path)
            
            modifications = self._generate_bug_fix_modifications(file_info, language, errors)
            if modifications:
                file_modifications.append({
                    "file_path": file_path,
                    "language": language,
                    "modifications": modifications,
                    "rationale": f"Fix potential issues based on error analysis"
                })
        
        return {
            "fixes": fixes,
            "file_modifications": file_modifications
        }
    
    def _generate_feature_implementation(self, issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation suggestions for feature requests."""
        fixes = []
        new_files = []
        file_modifications = []
        
        features = issue_analysis["extracted_info"].get("features_requested", [])
        relevant_files = issue_analysis["analysis"]["relevant_files"]
        
        # Analyze requested features
        for feature in features[:2]:  # Limit to top 2 features
            feature_plan = self._plan_feature_implementation(feature, relevant_files)
            
            if feature_plan.get("new_files"):
                new_files.extend(feature_plan["new_files"])
            
            if feature_plan.get("modifications"):
                file_modifications.extend(feature_plan["modifications"])
            
            if feature_plan.get("suggestions"):
                fixes.extend(feature_plan["suggestions"])
        
        return {
            "fixes": fixes,
            "new_files": new_files,
            "file_modifications": file_modifications
        }
    
    def _generate_performance_improvements(self, issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance improvement suggestions."""
        improvements = []
        file_modifications = []
        
        relevant_files = issue_analysis["analysis"]["relevant_files"]
        
        # Analyze code for common performance issues
        for file_info in relevant_files[:5]:
            file_path = file_info["file_path"]
            language = self._detect_language(file_path)
            content = file_info.get("content_preview", "")
            
            perf_improvements = self._analyze_performance_patterns(content, language)
            if perf_improvements:
                file_modifications.append({
                    "file_path": file_path,
                    "language": language,
                    "modifications": perf_improvements,
                    "rationale": "Performance optimization suggestions"
                })
        
        return {
            "improvements": improvements,
            "file_modifications": file_modifications
        }
    
    def _analyze_error_pattern(self, error: str) -> List[Dict[str, Any]]:
        """Analyze an error pattern and suggest fixes."""
        fixes = []
        
        # Common error patterns and their fixes
        if "nullpointerexception" in error.lower() or "none" in error.lower():
            fixes.append({
                "type": "null_check",
                "description": "Add null/None checking to prevent null pointer exceptions",
                "priority": "high",
                "pattern": "null_safety"
            })
        
        if "indexerror" in error.lower() or "arrayindexoutofbounds" in error.lower():
            fixes.append({
                "type": "bounds_check",
                "description": "Add bounds checking for array/list access",
                "priority": "high",
                "pattern": "bounds_safety"
            })
        
        if "keyerror" in error.lower() or "nosuchelementexception" in error.lower():
            fixes.append({
                "type": "key_validation",
                "description": "Add key/element existence validation",
                "priority": "medium",
                "pattern": "key_safety"
            })
        
        if "connectionerror" in error.lower() or "timeout" in error.lower():
            fixes.append({
                "type": "retry_logic",
                "description": "Add retry logic and connection error handling",
                "priority": "medium",
                "pattern": "network_resilience"
            })
        
        return fixes
    
    def _generate_bug_fix_modifications(self, file_info: Dict[str, Any], 
                                      language: str, errors: List[str]) -> List[Dict[str, Any]]:
        """Generate specific code modifications for bug fixes."""
        modifications = []
        content = file_info.get("content_preview", "")
        
        # Error handling improvements
        if any("exception" in error.lower() for error in errors):
            if language == "python" and "try:" not in content:
                modifications.append({
                    "type": "add_error_handling",
                    "description": "Add try-catch error handling",
                    "code_template": self.templates["error_handling"]["python"],
                    "priority": "high"
                })
        
        # Null checking
        if any("none" in error.lower() or "null" in error.lower() for error in errors):
            modifications.append({
                "type": "add_null_checks",
                "description": "Add null/None validation",
                "code_template": self.templates["null_check"][language],
                "priority": "medium"
            })
        
        # Input validation
        if "validation" in " ".join(errors).lower():
            modifications.append({
                "type": "add_validation",
                "description": "Add input validation",
                "code_template": self.templates["validation"][language],
                "priority": "medium"
            })
        
        return modifications
    
    def _plan_feature_implementation(self, feature: str, 
                                   relevant_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan implementation for a requested feature."""
        plan = {
            "new_files": [],
            "modifications": [],
            "suggestions": []
        }
        
        # TODO: Use relevant_files to suggest modifications to existing files
        # instead of always creating new ones
        _ = relevant_files  # Currently unused but available for context
        
        feature_lower = feature.lower()
        
        # API endpoint feature
        if "api" in feature_lower or "endpoint" in feature_lower:
            plan["new_files"].append({
                "file_path": "src/api/new_endpoint.py",
                "description": f"New API endpoint for {feature}",
                "template": "api_endpoint",
                "priority": "high"
            })
            
            plan["suggestions"].append({
                "type": "api_design",
                "description": f"Design REST API interface for {feature}",
                "priority": "high"
            })
        
        # Database feature
        if "database" in feature_lower or "model" in feature_lower:
            plan["new_files"].append({
                "file_path": "src/models/new_model.py",
                "description": f"Database model for {feature}",
                "template": "database_model",
                "priority": "high"
            })
        
        # UI feature
        if "ui" in feature_lower or "interface" in feature_lower:
            plan["new_files"].append({
                "file_path": "src/components/NewComponent.jsx",
                "description": f"UI component for {feature}",
                "template": "react_component",
                "priority": "medium"
            })
        
        return plan
    
    def _analyze_performance_patterns(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Analyze code for performance improvement opportunities."""
        improvements = []
        
        # Common performance anti-patterns
        if language == "python":
            # Inefficient loops
            if re.search(r'for.*in.*range\(len\(', content):
                improvements.append({
                    "type": "loop_optimization",
                    "description": "Replace range(len()) with direct iteration",
                    "priority": "low"
                })
            
            # String concatenation in loops
            if re.search(r'for.*\+\s*=.*str', content):
                improvements.append({
                    "type": "string_optimization",
                    "description": "Use list.join() instead of string concatenation in loops",
                    "priority": "medium"
                })
        
        elif language == "javascript":
            # DOM queries in loops
            if re.search(r'for.*document\.', content):
                improvements.append({
                    "type": "dom_optimization",
                    "description": "Cache DOM queries outside loops",
                    "priority": "medium"
                })
        
        return improvements
    
    def _generate_general_improvements(self, relevant_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate general code quality improvements."""
        improvements = []
        
        for file_info in relevant_files[:3]:
            content = file_info.get("content_preview", "")
            language = self._detect_language(file_info["file_path"])
            
            # Code quality suggestions
            if language == "python":
                if "# TODO" in content or "# FIXME" in content:
                    improvements.append({
                        "type": "todo_cleanup",
                        "description": f"Address TODO/FIXME comments in {file_info['file_path']}",
                        "priority": "low"
                    })
                
                if not re.search(r'""".*"""', content) and "def " in content:
                    improvements.append({
                        "type": "documentation",
                        "description": f"Add docstrings to functions in {file_info['file_path']}",
                        "priority": "low"
                    })
        
        return improvements
    
    def _generate_test_suggestions(self, issue_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test suggestions for the issue."""
        suggestions = []
        
        issue_type = issue_analysis["extracted_info"]["issue_type"]
        relevant_files = issue_analysis["analysis"]["relevant_files"]
        
        if issue_type == "bug":
            suggestions.append({
                "type": "regression_test",
                "description": "Add regression test to prevent this bug from reoccurring",
                "priority": "high",
                "test_type": "unit"
            })
            
            suggestions.append({
                "type": "error_handling_test",
                "description": "Test error handling and edge cases",
                "priority": "medium",
                "test_type": "unit"
            })
        
        elif issue_type == "feature":
            suggestions.append({
                "type": "feature_test",
                "description": "Add comprehensive tests for new feature",
                "priority": "high",
                "test_type": "integration"
            })
        
        # Suggest tests for modified files
        for file_info in relevant_files[:3]:
            suggestions.append({
                "type": "file_coverage",
                "description": f"Ensure test coverage for {file_info['file_path']}",
                "priority": "medium",
                "test_type": "unit"
            })
        
        return suggestions
    
    def _perform_safety_checks(self, suggestions: Dict[str, Any]) -> List[str]:
        """Perform safety checks on generated suggestions."""
        warnings = []
        
        # Check file count
        total_files = len(suggestions.get("file_modifications", [])) + len(suggestions.get("new_files", []))
        if total_files > self.max_files_per_pr:
            warnings.append(f"High number of files to modify ({total_files}). Consider breaking into smaller changes.")
        
        # Check for blocked file patterns
        all_files = [mod["file_path"] for mod in suggestions.get("file_modifications", [])]
        all_files.extend([nf["file_path"] for nf in suggestions.get("new_files", [])])
        
        for file_path in all_files:
            for pattern in self.blocked_patterns:
                if re.match(pattern.replace("*", ".*"), file_path):
                    warnings.append(f"Blocked file pattern detected: {file_path} matches {pattern}")
        
        # Check for high-risk modifications
        high_risk_types = ["database_schema", "security", "authentication"]
        for fix in suggestions.get("fixes", []):
            if fix.get("type") in high_risk_types:
                warnings.append(f"High-risk modification detected: {fix['type']}")
        
        return warnings
    
    def _calculate_confidence(self, issue_analysis: Dict[str, Any]) -> str:
        """Calculate confidence level for generated suggestions."""
        confidence_score = issue_analysis["analysis"].get("confidence_score", 0.0)
        relevant_files_count = len(issue_analysis["analysis"].get("relevant_files", []))
        
        if confidence_score > 80 and relevant_files_count > 3:
            return "high"
        elif confidence_score > 50 and relevant_files_count > 1:
            return "medium"
        else:
            return "low"
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension = Path(file_path).suffix.lower()
        
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala"
        }
        
        return language_map.get(extension, "unknown")
    
    def format_suggestions_for_pr(self, suggestions: Dict[str, Any], 
                                 issue_number: int) -> Dict[str, str]:
        """Format suggestions for pull request creation."""
        title_template = self.config.get("pull_requests", {}).get("template", {}).get("title_prefix", "[Auto] ")
        body_template = self.config.get("pull_requests", {}).get("template", {}).get("body_template", "")
        
        # Generate PR title
        title = f"{title_template}Fix for issue #{issue_number}"
        
        # Generate PR body
        summary = self._generate_pr_summary(suggestions)
        changes = self._generate_changes_list(suggestions)
        
        body = body_template.format(
            summary=summary,
            changes=changes,
            issue_number=issue_number
        )
        
        return {
            "title": title,
            "body": body
        }
    
    def _generate_pr_summary(self, suggestions: Dict[str, Any]) -> str:
        """Generate a summary for the PR description."""
        fixes_count = len(suggestions.get("fixes", []))
        modifications_count = len(suggestions.get("file_modifications", []))
        new_files_count = len(suggestions.get("new_files", []))
        
        summary_parts = []
        
        if fixes_count > 0:
            summary_parts.append(f"{fixes_count} fix{'es' if fixes_count > 1 else ''}")
        
        if modifications_count > 0:
            summary_parts.append(f"{modifications_count} file modification{'s' if modifications_count > 1 else ''}")
        
        if new_files_count > 0:
            summary_parts.append(f"{new_files_count} new file{'s' if new_files_count > 1 else ''}")
        
        if summary_parts:
            return f"This PR implements {', '.join(summary_parts)} based on automated analysis."
        else:
            return "This PR contains suggested improvements based on automated analysis."
    
    def _generate_changes_list(self, suggestions: Dict[str, Any]) -> str:
        """Generate a list of changes for the PR description."""
        changes = []
        
        for fix in suggestions.get("fixes", []):
            changes.append(f"- {fix['description']}")
        
        for mod in suggestions.get("file_modifications", []):
            changes.append(f"- Modified `{mod['file_path']}`: {mod['rationale']}")
        
        for new_file in suggestions.get("new_files", []):
            changes.append(f"- Added `{new_file['file_path']}`: {new_file['description']}")
        
        return "\n".join(changes) if changes else "- Code analysis and improvements"


# Global generator instance
_code_generator = None


def get_code_generator() -> CodeGenerator:
    """Get or create global code generator instance."""
    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator()
    return _code_generator