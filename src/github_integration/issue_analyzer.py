"""
GitHub Issue Analyzer for Qdrant MCP RAG Server

Combines GitHub issue data with RAG search to provide intelligent issue analysis.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class IssueAnalyzer:
    """
    Analyzes GitHub issues using RAG search capabilities.
    
    Extracts relevant information from issues and searches the codebase
    for related code, patterns, and potential solutions.
    """
    
    def __init__(self, github_client, search_functions: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the issue analyzer.
        
        Args:
            github_client: GitHub client instance
            search_functions: Dictionary of RAG search functions (search, search_code, etc.)
            config: Optional configuration override
        """
        self.github_client = github_client
        self.search_functions = search_functions
        self.config = config or {}
        
        # Analysis configuration
        analysis_config = self.config.get("issues", {}).get("analysis", {})
        self.search_limit = analysis_config.get("search_limit", 10)
        self.context_expansion = analysis_config.get("context_expansion", True)
        self.include_dependencies = analysis_config.get("include_dependencies", True)
        self.similarity_threshold = analysis_config.get("code_similarity_threshold", 0.7)
        
        # Patterns for extracting information from issue text
        self.error_patterns = [
            r"error[:\s]+(.+)",
            r"exception[:\s]+(.+)",
            r"failed[:\s]+(.+)",
            r"traceback[:\s]+([\s\S]+?)(?:\n\n|\Z)",
            r"```(?:python|javascript|java|go|rust|cpp|c\+\+)?\n([\s\S]+?)\n```"
        ]
        
        self.code_patterns = [
            r"`([^`]+)`",  # Inline code
            r"```[\w]*\n([\s\S]+?)\n```",  # Code blocks
            r"file[:\s]+([^\s\n]+)",  # File references
            r"function[:\s]+([^\s\n\(]+)",  # Function references
            r"class[:\s]+([^\s\n\(]+)",  # Class references
        ]
        
        self.feature_patterns = [
            r"add[:\s]+(.+)",
            r"implement[:\s]+(.+)",
            r"create[:\s]+(.+)",
            r"feature[:\s]+(.+)",
            r"enhancement[:\s]+(.+)"
        ]
    
    def analyze_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a GitHub issue.
        
        Args:
            issue_number: Issue number to analyze
            
        Returns:
            Analysis results with search results and recommendations
        """
        try:
            # Get issue details
            issue = self.github_client.get_issue(issue_number)
            
            # Extract information from issue
            extracted_info = self._extract_issue_information(issue)
            
            # Perform RAG searches
            search_results = self._perform_rag_searches(extracted_info)
            
            # Analyze search results
            analysis = self._analyze_search_results(search_results, extracted_info)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issue, extracted_info, analysis)
            
            # Check configuration for response verbosity
            analysis_config = self.config.get("issues", {}).get("analysis", {})
            response_verbosity = analysis_config.get("response_verbosity", "full")
            include_raw_search = analysis_config.get("include_raw_search_results", True)
            
            result = {
                "issue": {
                    "number": issue_number,
                    "title": issue["title"],
                    "body": issue["body"][:500] + "..." if len(issue["body"]) > 500 else issue["body"],
                    "labels": issue["labels"],
                    "state": issue["state"]
                },
                "extracted_info": self._summarize_extracted_info(extracted_info) if response_verbosity == "summary" else extracted_info,
                "analysis": analysis,
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }
            
            # Include search results based on configuration
            if include_raw_search and response_verbosity != "summary":
                result["search_results"] = search_results
            elif response_verbosity == "summary":
                # Create a summarized version
                result["search_summary"] = self._summarize_search_results(search_results, analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze issue {issue_number}: {e}")
            raise
    
    def _extract_issue_information(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from issue text."""
        title = issue["title"]
        body = issue["body"] or ""
        combined_text = f"{title}\n{body}"
        
        # Include comments in analysis
        comments_text = ""
        if issue.get("comments"):
            comments_text = "\n".join([comment["body"] for comment in issue["comments"]])
            combined_text += f"\n{comments_text}"
        
        extracted = {
            "title": title,
            "body": body,
            "full_text": combined_text,
            "errors": [],
            "code_snippets": [],
            "file_references": [],
            "function_references": [],
            "class_references": [],
            "features_requested": [],
            "keywords": [],
            "issue_type": self._classify_issue_type(title, body),
            "priority": self._estimate_priority(issue),
            "complexity": self._estimate_complexity(combined_text)
        }
        
        # Extract errors and stack traces
        for pattern in self.error_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            extracted["errors"].extend(matches)
        
        # Extract code references
        for pattern in self.code_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            if "file" in pattern:
                extracted["file_references"].extend(matches)
            elif "function" in pattern:
                extracted["function_references"].extend(matches)
            elif "class" in pattern:
                extracted["class_references"].extend(matches)
            else:
                extracted["code_snippets"].extend(matches)
        
        # Extract feature requests
        for pattern in self.feature_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            extracted["features_requested"].extend(matches)
        
        # Extract keywords (simple approach - can be enhanced with NLP)
        keywords = self._extract_keywords(combined_text)
        extracted["keywords"] = keywords
        
        return extracted
    
    def _classify_issue_type(self, title: str, body: str) -> str:
        """Classify the type of issue."""
        combined = f"{title} {body}".lower()
        
        bug_indicators = ["bug", "error", "exception", "crash", "fail", "broken", "issue", "problem"]
        feature_indicators = ["feature", "enhancement", "add", "implement", "create", "support"]
        doc_indicators = ["documentation", "docs", "readme", "guide", "tutorial", "example"]
        performance_indicators = ["slow", "performance", "speed", "optimize", "memory", "cpu"]
        
        if any(indicator in combined for indicator in bug_indicators):
            return "bug"
        elif any(indicator in combined for indicator in feature_indicators):
            return "feature"
        elif any(indicator in combined for indicator in doc_indicators):
            return "documentation"
        elif any(indicator in combined for indicator in performance_indicators):
            return "performance"
        else:
            return "unknown"
    
    def _estimate_priority(self, issue: Dict[str, Any]) -> str:
        """Estimate issue priority based on labels and content."""
        labels = [label.lower() for label in issue.get("labels", [])]
        
        if any(label in ["critical", "urgent", "blocker", "p0"] for label in labels):
            return "critical"
        elif any(label in ["high", "important", "p1"] for label in labels):
            return "high"
        elif any(label in ["low", "minor", "p3"] for label in labels):
            return "low"
        else:
            return "medium"
    
    def _estimate_complexity(self, text: str) -> str:
        """Estimate implementation complexity."""
        # Simple heuristic based on text content
        if len(text) > 2000:
            return "high"
        elif len(text) > 500:
            return "medium"
        else:
            return "low"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Remove common words and extract technical terms
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        # Simple word extraction (can be enhanced with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in common_words]
        
        # Get most frequent keywords
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:20]]
    
    def _perform_rag_searches(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform various RAG searches based on extracted information."""
        search_results = {
            "code_search": [],
            "docs_search": [],
            "general_search": [],
            "file_search": [],
            "error_search": []
        }
        
        # Determine context level based on issue type
        issue_type = extracted_info.get("issue_type", "unknown")
        context_level = {
            "bug": "method",        # Bugs need implementation details
            "feature": "class",     # Features need structure overview
            "documentation": "file", # Docs need high-level understanding
            "performance": "method", # Performance issues need implementation details
            "unknown": "class"      # Default to class level
        }.get(issue_type, "class")
        
        # Get progressive context configuration
        analysis_config = self.config.get("issues", {}).get("analysis", {})
        progressive_config = analysis_config.get("progressive_context", {})
        progressive_enabled = progressive_config.get("enabled", True)
        
        # Prepare search queries
        queries = []
        
        # Add title and keywords as primary queries
        if extracted_info["title"]:
            queries.append(("title", extracted_info["title"]))
        
        # Add error messages
        for error in extracted_info["errors"][:3]:  # Limit to top 3 errors
            if error.strip():
                queries.append(("error", error.strip()))
        
        # Add function/class references
        for func in extracted_info["function_references"][:3]:
            queries.append(("function", func))
        
        for cls in extracted_info["class_references"][:3]:
            queries.append(("class", cls))
        
        # Add feature keywords
        for feature in extracted_info["features_requested"][:2]:
            queries.append(("feature", feature))
        
        # Add top keywords
        for keyword in extracted_info["keywords"][:5]:
            queries.append(("keyword", keyword))
        
        # Deduplicate queries
        seen_queries = set()
        unique_queries = []
        
        for query_type, query_text in queries:
            normalized = query_text.lower().strip()
            if normalized not in seen_queries and len(normalized) > 3:
                seen_queries.add(normalized)
                unique_queries.append((query_type, query_text))
        
        # Limit total queries
        unique_queries = unique_queries[:8]  # Max 8 unique queries
        
        # Perform searches with progressive context
        for query_type, query_text in unique_queries:
            try:
                # Base search parameters
                search_params = {
                    "query": query_text,
                    "n_results": self.search_limit if query_type != "keyword" else self.search_limit // 2,
                    "include_context": self.context_expansion
                }
                
                # Add progressive context parameters if enabled
                if progressive_enabled:
                    search_params.update({
                        "progressive_mode": True,
                        "context_level": context_level,
                        "include_expansion_options": True,
                        "semantic_cache": True
                    })
                
                if query_type in ["function", "class", "error"]:
                    # Use code search for technical queries
                    search_params["include_dependencies"] = self.include_dependencies
                    results = self.search_functions["search_code"](**search_params)
                    search_results["code_search"].extend(results.get("results", []))
                
                elif query_type == "feature":
                    # Use docs search for feature requests to find related documentation
                    results = self.search_functions.get("search_docs", self.search_functions["search"])(**search_params)
                    search_results["docs_search"].extend(results.get("results", []))
                
                else:
                    # For keywords and title, try code search first as it's most likely
                    # to find relevant implementation details
                    try:
                        results = self.search_functions["search_code"](**search_params)
                        search_results["code_search"].extend(results.get("results", []))
                    except Exception as e:
                        # If code search fails, try docs search as fallback
                        logger.debug(f"Code search failed for '{query_text}', trying docs: {e}")
                        try:
                            results = self.search_functions.get("search_docs", self.search_functions["search"])(**search_params)
                            search_results["docs_search"].extend(results.get("results", []))
                        except Exception as e2:
                            logger.warning(f"All searches failed for query '{query_text}': {e2}")
                    
            except Exception as e:
                logger.warning(f"Search failed for query '{query_text}': {e}")
        
        # Remove duplicates and sort by score
        for category in search_results:
            seen_files = set()
            unique_results = []
            
            for result in search_results[category]:
                file_path = result.get("file_path", "")
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    unique_results.append(result)
            
            # Sort by score and limit results
            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            search_results[category] = unique_results[:self.search_limit]
        
        return search_results
    
    def _analyze_search_results(self, search_results: Dict[str, Any], 
                               extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search results to identify patterns and relevant files."""
        analysis = {
            "relevant_files": [],
            "potential_related_issues": [],
            "code_patterns": [],
            "suggested_investigation_areas": [],
            "confidence_score": 0.0
        }
        
        # Collect all results
        all_results = []
        for category, results in search_results.items():
            all_results.extend(results)
        
        # Analyze file relevance
        file_scores = {}
        for result in all_results:
            file_path = result.get("file_path", "")
            score = result.get("score", 0)
            
            if file_path:
                if file_path in file_scores:
                    file_scores[file_path] = max(file_scores[file_path], score)
                else:
                    file_scores[file_path] = score
        
        # Sort files by relevance
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter files above similarity threshold
        relevant_files = []
        for file_path, score in sorted_files:
            if score >= self.similarity_threshold:
                file_info = next((r for r in all_results if r.get("file_path") == file_path), {})
                relevant_files.append({
                    "file_path": file_path,
                    "relevance_score": score,
                    "language": file_info.get("language"),
                    "content_preview": file_info.get("content", "")[:200] + "..." if file_info.get("content") else ""
                })
        
        analysis["relevant_files"] = relevant_files[:10]  # Top 10 most relevant files
        
        # Extract code patterns
        patterns = set()
        for result in all_results:
            content = result.get("content", "")
            # Look for common patterns (imports, function calls, etc.)
            imports = re.findall(r'(?:import|from)\s+([^\s\n]+)', content)
            patterns.update(imports)
            
            function_calls = re.findall(r'(\w+)\s*\(', content)
            patterns.update(function_calls[:5])  # Limit to avoid noise
        
        analysis["code_patterns"] = list(patterns)[:20]  # Top 20 patterns
        
        # Calculate confidence score
        total_results = len(all_results)
        high_score_results = len([r for r in all_results if r.get("score", 0) >= self.similarity_threshold])
        
        if total_results > 0:
            analysis["confidence_score"] = (high_score_results / total_results) * 100
        
        # Generate investigation suggestions
        suggestions = []
        
        if extracted_info["issue_type"] == "bug" and extracted_info["errors"]:
            suggestions.append("Investigate error handling in related code sections")
            suggestions.append("Check for similar error patterns in the codebase")
        
        if extracted_info["issue_type"] == "feature":
            suggestions.append("Look for similar implementations in the codebase")
            suggestions.append("Check if related functionality already exists")
        
        if relevant_files:
            suggestions.append(f"Focus investigation on {len(relevant_files)} most relevant files")
        
        if analysis["code_patterns"]:
            suggestions.append("Examine common code patterns and dependencies")
        
        analysis["suggested_investigation_areas"] = suggestions
        
        return analysis
    
    def _generate_recommendations(self, issue: Dict[str, Any], 
                                 extracted_info: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis."""
        recommendations = {
            "immediate_actions": [],
            "investigation_steps": [],
            "potential_solutions": [],
            "related_files_to_examine": [],
            "estimated_effort": "unknown",
            "risk_level": "medium"
        }
        
        # Immediate actions based on issue type
        if extracted_info["issue_type"] == "bug":
            recommendations["immediate_actions"].extend([
                "Reproduce the issue locally",
                "Check recent changes to related files",
                "Review error logs and stack traces"
            ])
            
            if extracted_info["errors"]:
                recommendations["immediate_actions"].append("Analyze specific error messages")
        
        elif extracted_info["issue_type"] == "feature":
            recommendations["immediate_actions"].extend([
                "Review existing similar functionality",
                "Assess impact on current architecture",
                "Create implementation plan"
            ])
        
        # Investigation steps
        if analysis["relevant_files"]:
            recommendations["investigation_steps"].append(
                f"Examine {len(analysis['relevant_files'])} most relevant files"
            )
            recommendations["related_files_to_examine"] = [
                f["file_path"] for f in analysis["relevant_files"][:5]
            ]
        
        if analysis["code_patterns"]:
            recommendations["investigation_steps"].append(
                "Analyze common code patterns and dependencies"
            )
        
        # Potential solutions based on analysis
        if analysis["confidence_score"] > 80:
            recommendations["potential_solutions"].append(
                "High confidence match found - likely similar issue resolved before"
            )
        elif analysis["confidence_score"] > 50:
            recommendations["potential_solutions"].append(
                "Moderate confidence - related code patterns identified"
            )
        else:
            recommendations["potential_solutions"].append(
                "Low confidence - may require broader investigation"
            )
        
        # Estimate effort and risk
        complexity = extracted_info["complexity"]
        priority = extracted_info["priority"]
        
        if complexity == "high" or priority == "critical":
            recommendations["estimated_effort"] = "high"
            recommendations["risk_level"] = "high"
        elif complexity == "medium" or priority == "high":
            recommendations["estimated_effort"] = "medium"
            recommendations["risk_level"] = "medium"
        else:
            recommendations["estimated_effort"] = "low"
            recommendations["risk_level"] = "low"
        
        return recommendations
    
    def _summarize_search_results(self, search_results: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summarized version of search results for reduced token consumption."""
        summary = {
            "total_searches": 0,
            "total_results": 0,
            "high_relevance_count": 0,
            "search_categories": {},
            "top_files": [],
            "key_insights": []
        }
        
        # Count searches and results
        for category, results in search_results.items():
            if results:
                summary["total_searches"] += 1
                summary["total_results"] += len(results)
                summary["search_categories"][category] = len(results)
                
                # Count high relevance results
                high_relevance = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
                summary["high_relevance_count"] += len(high_relevance)
        
        # Get top files from analysis
        if analysis.get("relevant_files"):
            max_files = self.config.get("issues", {}).get("analysis", {}).get("max_relevant_files", 5)
            summary["top_files"] = [
                {
                    "path": f["file_path"],
                    "score": round(f["relevance_score"], 3),
                    "language": f.get("language", "unknown")
                }
                for f in analysis["relevant_files"][:max_files]
            ]
        
        # Generate key insights
        if summary["high_relevance_count"] > 5:
            summary["key_insights"].append(f"Found {summary['high_relevance_count']} highly relevant code sections")
        
        if analysis.get("code_patterns"):
            summary["key_insights"].append(f"Identified {len(analysis['code_patterns'])} common code patterns")
        
        if analysis.get("confidence_score", 0) > 70:
            summary["key_insights"].append("High confidence match - similar issues likely resolved before")
        
        return summary
    
    def _summarize_extracted_info(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summarized version of extracted information."""
        summary = {
            "issue_type": extracted_info["issue_type"],
            "priority": extracted_info["priority"],
            "complexity": extracted_info["complexity"],
            "key_references": {
                "errors": len(extracted_info.get("errors", [])),
                "files": len(extracted_info.get("file_references", [])),
                "functions": len(extracted_info.get("function_references", [])),
                "classes": len(extracted_info.get("class_references", [])),
                "features": len(extracted_info.get("features_requested", []))
            },
            "top_keywords": extracted_info.get("keywords", [])[:5]
        }
        
        # Include sample references if present
        if extracted_info.get("errors"):
            summary["sample_error"] = extracted_info["errors"][0][:100] + "..." if len(extracted_info["errors"][0]) > 100 else extracted_info["errors"][0]
        
        if extracted_info.get("file_references"):
            summary["sample_files"] = extracted_info["file_references"][:3]
        
        return summary


# Global analyzer instance
_issue_analyzer = None


def reset_issue_analyzer():
    """Reset the global issue analyzer instance to force reload of configuration."""
    global _issue_analyzer
    _issue_analyzer = None


def get_issue_analyzer(github_client, search_functions: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> IssueAnalyzer:
    """Get or create global issue analyzer instance.
    
    Args:
        github_client: GitHub client instance
        search_functions: Dictionary of RAG search functions
        config: Optional configuration to pass to IssueAnalyzer
    
    Returns:
        IssueAnalyzer instance
    """
    global _issue_analyzer
    if _issue_analyzer is None:
        _issue_analyzer = IssueAnalyzer(github_client, search_functions, config)
    elif config is not None:
        # Update configuration if provided
        _issue_analyzer.config = config
        # Update analysis configuration
        analysis_config = config.get("issues", {}).get("analysis", {})
        _issue_analyzer.search_limit = analysis_config.get("search_limit", 10)
        _issue_analyzer.context_expansion = analysis_config.get("context_expansion", True)
        _issue_analyzer.include_dependencies = analysis_config.get("include_dependencies", True)
        _issue_analyzer.similarity_threshold = analysis_config.get("code_similarity_threshold", 0.7)
    return _issue_analyzer