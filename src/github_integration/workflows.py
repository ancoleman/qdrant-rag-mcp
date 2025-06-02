"""
GitHub Workflows for Issue Resolution

Orchestrates end-to-end workflows for GitHub issue analysis and resolution.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GitHubWorkflows:
    """
    Orchestrates complete GitHub issue resolution workflows.
    
    Combines GitHub client, issue analysis, code generation, and safety checks
    to provide comprehensive issue resolution capabilities.
    """
    
    def __init__(self, github_client, issue_analyzer, code_generator, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GitHub workflows.
        
        Args:
            github_client: GitHub API client
            issue_analyzer: Issue analysis engine
            code_generator: Code generation engine
            config: Optional configuration override
        """
        self.github_client = github_client
        self.issue_analyzer = issue_analyzer
        self.code_generator = code_generator
        self.config = config or {}
        
        # Workflow configuration
        self.workflows_config = self.config.get("workflows", {})
        self.safety_config = self.config.get("safety", {})
        
        # Safety settings
        self.dry_run_by_default = self.safety_config.get("dry_run_by_default", True)
        self.require_confirmation = self.safety_config.get("require_confirmation", True)
        self.audit_logging = self.safety_config.get("audit_logging", True)
    
    def analyze_issue_workflow(self, issue_number: int) -> Dict[str, Any]:
        """
        Complete issue analysis workflow.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info(f"Starting issue analysis workflow for issue #{issue_number}")
            
            # Ensure repository is set
            if not self.github_client.get_current_repository():
                raise ValueError("No repository context set. Use github_switch_repository first.")
            
            # Perform issue analysis
            analysis_result = self.issue_analyzer.analyze_issue(issue_number)
            
            # Add workflow metadata
            workflow_result = {
                "workflow_type": "issue_analysis",
                "issue_number": issue_number,
                "repository": self.github_client.get_current_repository().full_name,
                "analysis": analysis_result,
                "workflow_status": "completed",
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            }
            
            # Log audit information
            if self.audit_logging:
                self._log_audit_event("issue_analysis", workflow_result)
            
            logger.info(f"Issue analysis workflow completed for issue #{issue_number}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Issue analysis workflow failed for issue #{issue_number}: {e}")
            return {
                "workflow_type": "issue_analysis",
                "issue_number": issue_number,
                "workflow_status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
    
    def suggest_fix_workflow(self, issue_number: int) -> Dict[str, Any]:
        """
        Generate fix suggestions workflow.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Fix suggestions and implementation plan
        """
        try:
            logger.info(f"Starting fix suggestion workflow for issue #{issue_number}")
            
            # First, analyze the issue
            analysis_result = self.analyze_issue_workflow(issue_number)
            
            if analysis_result["workflow_status"] != "completed":
                return analysis_result  # Return error from analysis
            
            # Generate fix suggestions
            issue_analysis = analysis_result["analysis"]
            suggestions = self.code_generator.generate_fix_suggestions(issue_analysis)
            
            # Evaluate feasibility
            feasibility = self._evaluate_fix_feasibility(suggestions, issue_analysis)
            
            # Create workflow result
            workflow_result = {
                "workflow_type": "fix_suggestion",
                "issue_number": issue_number,
                "repository": self.github_client.get_current_repository().full_name,
                "analysis": issue_analysis,
                "suggestions": suggestions,
                "feasibility": feasibility,
                "workflow_status": "completed",
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            }
            
            # Log audit information
            if self.audit_logging:
                self._log_audit_event("fix_suggestion", workflow_result)
            
            logger.info(f"Fix suggestion workflow completed for issue #{issue_number}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Fix suggestion workflow failed for issue #{issue_number}: {e}")
            return {
                "workflow_type": "fix_suggestion",
                "issue_number": issue_number,
                "workflow_status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
    
    def resolve_issue_workflow(self, issue_number: int, dry_run: Optional[bool] = None) -> Dict[str, Any]:
        """
        Complete issue resolution workflow with PR creation.
        
        Args:
            issue_number: GitHub issue number
            dry_run: Override default dry-run setting
            
        Returns:
            Resolution workflow results
        """
        try:
            # Determine if this should be a dry run
            is_dry_run = dry_run if dry_run is not None else self.dry_run_by_default
            
            logger.info(f"Starting issue resolution workflow for issue #{issue_number} (dry_run={is_dry_run})")
            
            # Check if auto-resolution is enabled
            auto_resolve_config = self.workflows_config.get("auto_resolve", {})
            if not auto_resolve_config.get("enabled", False) and not is_dry_run:
                return {
                    "workflow_type": "issue_resolution",
                    "issue_number": issue_number,
                    "workflow_status": "blocked",
                    "message": "Auto-resolution is disabled. Use dry_run=True for suggestions only.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Generate fix suggestions
            suggestion_result = self.suggest_fix_workflow(issue_number)
            
            if suggestion_result["workflow_status"] != "completed":
                return suggestion_result  # Return error from suggestion workflow
            
            # Evaluate if issue is suitable for auto-resolution
            suitability = self._evaluate_auto_resolution_suitability(suggestion_result)
            
            if not suitability["suitable"] and not is_dry_run:
                return {
                    "workflow_type": "issue_resolution",
                    "issue_number": issue_number,
                    "workflow_status": "not_suitable",
                    "suitability": suitability,
                    "suggestions": suggestion_result["suggestions"],
                    "message": "Issue not suitable for auto-resolution. Consider manual review.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Create workflow result
            workflow_result = {
                "workflow_type": "issue_resolution",
                "issue_number": issue_number,
                "repository": self.github_client.get_current_repository().full_name,
                "dry_run": is_dry_run,
                "suitability": suitability,
                "suggestion_result": suggestion_result,
                "workflow_status": "planning",
                "started_at": datetime.now().isoformat()
            }
            
            if is_dry_run:
                # Dry run - just return suggestions
                workflow_result.update({
                    "workflow_status": "dry_run_completed",
                    "message": "Dry run completed. Review suggestions before proceeding.",
                    "pr_preview": self._generate_pr_preview(suggestion_result, issue_number),
                    "completed_at": datetime.now().isoformat()
                })
            else:
                # Actual resolution - create PR (placeholder for now)
                pr_result = self._create_resolution_pr(suggestion_result, issue_number)
                workflow_result.update({
                    "pr_result": pr_result,
                    "workflow_status": "completed" if pr_result.get("success") else "failed",
                    "completed_at": datetime.now().isoformat()
                })
            
            # Log audit information
            if self.audit_logging:
                self._log_audit_event("issue_resolution", workflow_result)
            
            logger.info(f"Issue resolution workflow completed for issue #{issue_number}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Issue resolution workflow failed for issue #{issue_number}: {e}")
            return {
                "workflow_type": "issue_resolution",
                "issue_number": issue_number,
                "workflow_status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
    
    def _evaluate_fix_feasibility(self, suggestions: Dict[str, Any], 
                                 issue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the feasibility of implementing suggested fixes."""
        feasibility = {
            "overall_score": 0.0,
            "factors": {},
            "recommendations": [],
            "risks": []
        }
        
        # Evaluate based on confidence level
        confidence = suggestions.get("confidence_level", "low")
        if confidence == "high":
            feasibility["factors"]["confidence"] = 0.8
        elif confidence == "medium":
            feasibility["factors"]["confidence"] = 0.6
        else:
            feasibility["factors"]["confidence"] = 0.3
        
        # Evaluate based on number of files to modify
        file_count = len(suggestions.get("file_modifications", [])) + len(suggestions.get("new_files", []))
        if file_count <= 3:
            feasibility["factors"]["complexity"] = 0.8
        elif file_count <= 7:
            feasibility["factors"]["complexity"] = 0.6
        else:
            feasibility["factors"]["complexity"] = 0.3
        
        # Evaluate based on safety warnings
        warning_count = len(suggestions.get("safety_warnings", []))
        if warning_count == 0:
            feasibility["factors"]["safety"] = 0.9
        elif warning_count <= 2:
            feasibility["factors"]["safety"] = 0.6
        else:
            feasibility["factors"]["safety"] = 0.2
        
        # Calculate overall score
        factors = feasibility["factors"]
        feasibility["overall_score"] = sum(factors.values()) / len(factors) if factors else 0.0
        
        # Generate recommendations
        if feasibility["overall_score"] >= 0.7:
            feasibility["recommendations"].append("High feasibility - suitable for auto-resolution")
        elif feasibility["overall_score"] >= 0.5:
            feasibility["recommendations"].append("Medium feasibility - review before implementing")
        else:
            feasibility["recommendations"].append("Low feasibility - manual review required")
        
        # Identify risks
        if warning_count > 0:
            feasibility["risks"].append(f"{warning_count} safety warning(s) detected")
        
        if file_count > 5:
            feasibility["risks"].append("High number of files to modify")
        
        if confidence == "low":
            feasibility["risks"].append("Low confidence in suggested fixes")
        
        return feasibility
    
    def _evaluate_auto_resolution_suitability(self, suggestion_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if an issue is suitable for automatic resolution."""
        suitability = {
            "suitable": False,
            "score": 0.0,
            "factors": {},
            "blockers": []
        }
        
        suggestions = suggestion_result["suggestions"]
        feasibility = suggestion_result["feasibility"]
        issue_analysis = suggestion_result["analysis"]
        
        # Check configuration thresholds
        auto_config = self.workflows_config.get("auto_resolve", {})
        confidence_threshold = auto_config.get("confidence_threshold", 0.85)
        max_complexity = auto_config.get("max_complexity", "medium")
        require_tests = auto_config.get("require_tests", True)
        
        # Evaluate confidence
        if feasibility["overall_score"] >= confidence_threshold:
            suitability["factors"]["confidence"] = 1.0
        else:
            suitability["factors"]["confidence"] = 0.0
            suitability["blockers"].append(f"Confidence below threshold ({feasibility['overall_score']:.2f} < {confidence_threshold})")
        
        # Evaluate complexity
        complexity = issue_analysis["extracted_info"]["complexity"]
        complexity_scores = {"low": 1.0, "medium": 0.7, "high": 0.3}
        max_complexity_scores = {"low": 1.0, "medium": 0.7, "high": 1.0}
        
        if complexity_scores.get(complexity, 0) >= complexity_scores.get(max_complexity, 0):
            suitability["factors"]["complexity"] = 1.0
        else:
            suitability["factors"]["complexity"] = 0.0
            suitability["blockers"].append(f"Complexity too high ({complexity} > {max_complexity})")
        
        # Check for safety warnings
        if suggestions.get("safety_warnings"):
            suitability["factors"]["safety"] = 0.0
            suitability["blockers"].append("Safety warnings present")
        else:
            suitability["factors"]["safety"] = 1.0
        
        # Check test requirements
        if require_tests and not suggestions.get("test_suggestions"):
            suitability["factors"]["tests"] = 0.0
            suitability["blockers"].append("No test suggestions provided")
        else:
            suitability["factors"]["tests"] = 1.0
        
        # Calculate overall suitability
        factors = suitability["factors"]
        suitability["score"] = sum(factors.values()) / len(factors) if factors else 0.0
        suitability["suitable"] = suitability["score"] >= 0.8 and len(suitability["blockers"]) == 0
        
        return suitability
    
    def _generate_pr_preview(self, suggestion_result: Dict[str, Any], issue_number: int) -> Dict[str, Any]:
        """Generate a preview of what the PR would look like."""
        suggestions = suggestion_result["suggestions"]
        
        # Generate PR title and body
        pr_format = self.code_generator.format_suggestions_for_pr(suggestions, issue_number)
        
        return {
            "title": pr_format["title"],
            "body": pr_format["body"],
            "files_to_modify": [mod["file_path"] for mod in suggestions.get("file_modifications", [])],
            "new_files": [nf["file_path"] for nf in suggestions.get("new_files", [])],
            "estimated_changes": len(suggestions.get("fixes", [])) + len(suggestions.get("improvements", [])),
            "branch_name": f"auto-fix-issue-{issue_number}",
            "draft": self.config.get("pull_requests", {}).get("draft_by_default", True)
        }
    
    def _create_resolution_pr(self, suggestion_result: Dict[str, Any], issue_number: int) -> Dict[str, Any]:
        """Create a pull request for issue resolution (placeholder implementation)."""
        # This is a placeholder - actual implementation would:
        # 1. Create a new branch
        # 2. Apply code changes
        # 3. Commit changes
        # 4. Create PR
        
        pr_preview = self._generate_pr_preview(suggestion_result, issue_number)
        
        # For now, return a placeholder result
        return {
            "success": False,
            "message": "PR creation not yet implemented - this is a preview",
            "preview": pr_preview,
            "next_steps": [
                "Implement branch creation",
                "Implement file modifications",
                "Implement commit and push",
                "Create actual PR via GitHub API"
            ]
        }
    
    def _log_audit_event(self, event_type: str, data: Dict[str, Any]):
        """Log audit event for compliance and debugging."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "repository": self.github_client.get_current_repository().full_name if self.github_client.get_current_repository() else "unknown",
            "data": data
        }
        
        # In a production system, this would write to an audit log
        logger.info(f"AUDIT: {event_type} - {audit_entry}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a running workflow (placeholder for future async workflows)."""
        return {
            "workflow_id": workflow_id,
            "status": "not_implemented",
            "message": "Workflow status tracking not yet implemented"
        }
    
    def list_recent_workflows(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent workflow executions (placeholder for future audit trail)."""
        return {
            "workflows": [],
            "message": "Workflow history tracking not yet implemented",
            "limit": limit
        }


# Global workflows instance
_github_workflows = None


def get_github_workflows(github_client, issue_analyzer, code_generator) -> GitHubWorkflows:
    """Get or create global GitHub workflows instance."""
    global _github_workflows
    if _github_workflows is None:
        _github_workflows = GitHubWorkflows(github_client, issue_analyzer, code_generator)
    return _github_workflows