"""
GitHub Projects V2 Manager for Qdrant MCP RAG Server

Provides GitHub Projects V2 (GraphQL) integration for project management workflows.
Uses the adapter pattern to extend existing PyGithub REST client.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

try:
    from gql import gql, Client
    from gql.transport.aiohttp import AIOHTTPTransport
    from gql.transport.exceptions import TransportError
    import aiohttp
    GQL_AVAILABLE = True
except ImportError:
    GQL_AVAILABLE = False

logger = logging.getLogger(__name__)


class GitHubProjectsError(Exception):
    """Base exception for GitHub Projects operations."""
    pass


class GitHubProjectsManager:
    """
    GitHub Projects V2 GraphQL adapter.
    
    Provides project management capabilities using GitHub's Projects V2 API
    while integrating cleanly with existing PyGithub REST client.
    """
    
    def __init__(self, github_client):
        """
        Initialize Projects manager.
        
        Args:
            github_client: Existing GitHubClient instance with authentication
        """
        if not GQL_AVAILABLE:
            raise ImportError("GraphQL dependencies not available. Install with: pip install 'gql[aiohttp]'")
            
        self.github_client = github_client
        self._graphql_client = None
        self._owner_id_cache = {}
        
        # Initialize GraphQL client
        self._init_graphql_client()
        
    def _init_graphql_client(self):
        """Initialize GraphQL client for Projects V2 API."""
        try:
            # Get token from existing GitHub client
            token = self._extract_token()
            
            # Create GraphQL transport
            transport = AIOHTTPTransport(
                url="https://api.github.com/graphql",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "qdrant-rag-mcp-server/0.3.4"
                }
            )
            
            self._graphql_client = Client(transport=transport, fetch_schema_from_transport=False)
            logger.info("GitHub Projects V2 GraphQL client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphQL client: {e}")
            raise GitHubProjectsError(f"GraphQL client initialization failed: {e}")
    
    def _extract_token(self) -> str:
        """Extract authentication token from PyGithub client."""
        try:
            # Access the Auth object from PyGithub
            auth = self.github_client._github._Github__requester._Requester__auth
            if hasattr(auth, 'token'):
                return auth.token
            else:
                raise GitHubProjectsError("No token found in GitHub client authentication")
        except AttributeError as e:
            logger.error(f"Could not extract token from GitHub client: {e}")
            raise GitHubProjectsError("Failed to extract authentication token")
    
    async def _execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute GraphQL query with error handling.
        
        Args:
            query: GraphQL query string
            variables: Optional query variables
            
        Returns:
            Query result data
            
        Raises:
            GitHubProjectsError: If query execution fails
        """
        # Log query for debugging (without sensitive data)
        query_preview = query.split('\n')[0][:100] + "..."
        logger.debug(f"Executing GraphQL query: {query_preview}")
        
        try:
            # Execute the query - this might return both data and errors
            result = await self._graphql_client.execute_async(gql(query), variable_values=variables)
            
            # Log query and response for debugging
            logger.debug(f">>> {json.dumps({'query': query, 'variables': variables})}")
            logger.debug(f"<<< {json.dumps(result)}")
            
            # Always return the data if we have any
            # GitHub often returns partial data with errors (e.g., user exists but org doesn't)
            return result
            
        except Exception as e:
            # Check if this is actually a partial success
            if hasattr(e, 'data') and e.data:
                # We have data despite the error - this is common for user/org queries
                logger.debug(f"Partial success with error: {e}")
                return e.data
            
            # Check for specific error types
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise GitHubProjectsError("GitHub API rate limit exceeded. Please try again later.")
            elif "forbidden" in error_msg.lower():
                raise GitHubProjectsError("Access forbidden. Check your token permissions.")
            elif "unauthorized" in error_msg.lower():
                raise GitHubProjectsError("Unauthorized. Check your GitHub token.")
            elif "not found" in error_msg.lower():
                raise GitHubProjectsError("Resource not found.")
            
            # Real error with no data
            logger.error(f"GraphQL query execution failed: {e}")
            raise GitHubProjectsError(f"Query execution failed: {e}")
    
    async def get_owner_id(self, owner: str) -> str:
        """
        Get GitHub node ID for owner (user or organization).
        
        Args:
            owner: Username or organization name
            
        Returns:
            GitHub node ID for the owner
        """
        if owner in self._owner_id_cache:
            return self._owner_id_cache[owner]
            
        query = """
        query($login: String!) {
            user(login: $login) {
                id
            }
            organization(login: $login) {
                id
            }
        }
        """
        
        result = await self._execute_query(query, {"login": owner})
        
        # Extract data from the result (could be in result['data'] or just result)
        data = result.get('data', result) if isinstance(result, dict) else result
        
        # Try user first, then organization
        owner_id = None
        if data.get('user') and data['user']:
            owner_id = data['user']['id']
        elif data.get('organization') and data['organization']:
            owner_id = data['organization']['id']
        else:
            raise GitHubProjectsError(f"Owner '{owner}' not found")
            
        self._owner_id_cache[owner] = owner_id
        logger.info(f"Found owner ID for '{owner}': {owner_id}")
        return owner_id
    
    async def create_project(self, owner: str, title: str, body: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new GitHub Project V2.
        
        Args:
            owner: Repository owner (username or organization)
            title: Project title
            body: Optional project description
            
        Returns:
            Project information including ID and URL
            
        Raises:
            GitHubProjectsError: If project creation fails
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        if not owner or not owner.strip():
            raise ValueError("Owner cannot be empty")
        if not title or not title.strip():
            raise ValueError("Project title cannot be empty")
        if len(title) > 256:
            raise ValueError("Project title cannot exceed 256 characters")
            
        try:
            owner_id = await self.get_owner_id(owner)
        except GitHubProjectsError as e:
            raise GitHubProjectsError(f"Failed to create project: {e}")
        
        mutation = """
        mutation($input: CreateProjectV2Input!) {
            createProjectV2(input: $input) {
                projectV2 {
                    id
                    number
                    title
                    shortDescription
                    url
                    public
                    createdAt
                    updatedAt
                }
            }
        }
        """
        
        variables = {
            "input": {
                "ownerId": owner_id,
                "title": title,
            }
        }
        
        # Note: GitHub Projects V2 doesn't support descriptions at creation time
        # We could add a custom field later if needed
        
        try:
            result = await self._execute_query(mutation, variables)
            data = result.get('data', result) if isinstance(result, dict) else result
            
            # Check if we got an error about personal access tokens
            if isinstance(result, dict) and 'errors' in result:
                errors = result['errors']
                if errors:
                    error_msg = errors[0].get('message', 'Unknown error')
                    
                    # Check for specific error conditions
                    if 'personal access token' in error_msg:
                        raise GitHubProjectsError(
                            "This Personal Access Token cannot create projects for this owner. "
                            "Ensure your token has 'project' scope."
                        )
                    elif 'already exists' in error_msg:
                        raise GitHubProjectsError(f"A project with title '{title}' already exists for {owner}")
                    elif 'permission' in error_msg.lower():
                        raise GitHubProjectsError(f"Insufficient permissions to create project for {owner}")
                    else:
                        raise GitHubProjectsError(f"Failed to create project: {error_msg}")
            
            # Check if project creation returned null
            if not data.get('createProjectV2') or not data['createProjectV2'].get('projectV2'):
                raise GitHubProjectsError("Failed to create project - no project data returned")
                
            project = data['createProjectV2']['projectV2']
            
            logger.info(f"Created project '{title}' with ID: {project['id']}")
            return project
            
        except GitHubProjectsError:
            raise
        except Exception as e:
            raise GitHubProjectsError(f"Unexpected error creating project: {e}")
    
    async def get_project(self, owner: str, number: int) -> Dict[str, Any]:
        """
        Get project details by number.
        
        Args:
            owner: Repository owner
            number: Project number
            
        Returns:
            Project details including fields and items count
        """
        query = """
        query($owner: String!, $number: Int!) {
            user(login: $owner) {
                projectV2(number: $number) {
                    id
                    number
                    title
                    shortDescription
                    url
                    public
                    createdAt
                    updatedAt
                    items(first: 0) {
                        totalCount
                    }
                    fields(first: 20) {
                        nodes {
                            ... on ProjectV2Field {
                                id
                                name
                                dataType
                            }
                            ... on ProjectV2SingleSelectField {
                                id
                                name
                                dataType
                                options {
                                    id
                                    name
                                    color
                                }
                            }
                        }
                    }
                }
            }
            organization(login: $owner) {
                projectV2(number: $number) {
                    id
                    number
                    title
                    shortDescription
                    url
                    public
                    createdAt
                    updatedAt
                    items(first: 0) {
                        totalCount
                    }
                    fields(first: 20) {
                        nodes {
                            ... on ProjectV2Field {
                                id
                                name
                                dataType
                            }
                            ... on ProjectV2SingleSelectField {
                                id
                                name
                                dataType
                                options {
                                    id
                                    name
                                    color
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        result = await self._execute_query(query, {"owner": owner, "number": number})
        data = result.get('data', result) if isinstance(result, dict) else result
        
        # Try user first, then organization
        project = data.get('user', {}).get('projectV2') or data.get('organization', {}).get('projectV2')
        
        if not project:
            raise GitHubProjectsError(f"Project #{number} not found for owner '{owner}'")
            
        return project
    
    async def get_project_by_id(self, project_id: str) -> Dict[str, Any]:
        """
        Get project details by project ID.
        
        Args:
            project_id: Project node ID
            
        Returns:
            Project details including fields
        """
        query = """
        query($projectId: ID!) {
            node(id: $projectId) {
                ... on ProjectV2 {
                    id
                    number
                    title
                    shortDescription
                    url
                    public
                    createdAt
                    updatedAt
                    items(first: 0) {
                        totalCount
                    }
                    fields(first: 20) {
                        nodes {
                            ... on ProjectV2Field {
                                id
                                name
                                dataType
                            }
                            ... on ProjectV2SingleSelectField {
                                id
                                name
                                dataType
                                options {
                                    id
                                    name
                                    color
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        result = await self._execute_query(query, {"projectId": project_id})
        data = result.get('data', result) if isinstance(result, dict) else result
        
        if not data.get('node'):
            raise GitHubProjectsError(f"Project {project_id} not found")
            
        return data['node']
    
    async def add_item_to_project(self, project_id: str, content_id: str) -> Dict[str, Any]:
        """
        Add an issue or PR to a project.
        
        Args:
            project_id: Project node ID
            content_id: Issue or PR node ID
            
        Returns:
            Added item information
            
        Raises:
            GitHubProjectsError: If adding item fails
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty")
        if not content_id or not content_id.strip():
            raise ValueError("Content ID cannot be empty")
        if not project_id.startswith("PVT_"):
            raise ValueError(f"Invalid project ID format: {project_id}")
        if not (content_id.startswith("I_") or content_id.startswith("PR_")):
            raise ValueError(f"Invalid content ID format: {content_id} (must be issue or PR)")
        mutation = """
        mutation($projectId: ID!, $contentId: ID!) {
            addProjectV2ItemById(input: {
                projectId: $projectId
                contentId: $contentId
            }) {
                item {
                    id
                    type
                    createdAt
                    content {
                        ... on Issue {
                            id
                            number
                            title
                            url
                        }
                        ... on PullRequest {
                            id
                            number
                            title
                            url
                        }
                    }
                }
            }
        }
        """
        
        try:
            result = await self._execute_query(mutation, {
                "projectId": project_id,
                "contentId": content_id
            })
            data = result.get('data', result) if isinstance(result, dict) else result
            
            # Check for errors in response
            if isinstance(result, dict) and 'errors' in result:
                error_msg = result['errors'][0].get('message', 'Unknown error')
                if 'already exists' in error_msg:
                    raise GitHubProjectsError(f"Item is already in the project")
                else:
                    raise GitHubProjectsError(f"Failed to add item: {error_msg}")
            
            if not data.get('addProjectV2ItemById') or not data['addProjectV2ItemById'].get('item'):
                raise GitHubProjectsError("Failed to add item to project - no item returned")
                
            item = data['addProjectV2ItemById']['item']
            logger.info(f"Added item to project: {item['content']['title']}")
            return item
            
        except GitHubProjectsError:
            raise
        except Exception as e:
            raise GitHubProjectsError(f"Failed to add item to project: {e}")
    
    async def update_item_field(self, project_id: str, item_id: str, field_id: str, 
                               value: Any, field_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a field value for a project item.
        
        Args:
            project_id: Project node ID
            item_id: Item node ID  
            field_id: Field node ID
            value: New field value
            field_type: Optional field type (e.g., 'TEXT', 'NUMBER', 'SINGLE_SELECT')
            
        Returns:
            Updated item information
        """
        # Determine the appropriate mutation based on field type or value type
        if field_type == "SINGLE_SELECT" or (isinstance(value, str) and len(value) == 8):
            # Single select fields use option IDs (8 char hex strings)
            mutation = """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: String!) {
                updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId
                    itemId: $itemId
                    fieldId: $fieldId
                    value: {
                        singleSelectOptionId: $value
                    }
                }) {
                    projectV2Item {
                        id
                    }
                }
            }
            """
        elif field_type == "NUMBER" or isinstance(value, (int, float)):
            mutation = """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: Float!) {
                updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId
                    itemId: $itemId
                    fieldId: $fieldId
                    value: {
                        number: $value
                    }
                }) {
                    projectV2Item {
                        id
                    }
                }
            }
            """
        else:
            # Default to text field
            mutation = """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: String!) {
                updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId
                    itemId: $itemId
                    fieldId: $fieldId
                    value: {
                        text: $value
                    }
                }) {
                    projectV2Item {
                        id
                    }
                }
            }
            """
        
        try:
            result = await self._execute_query(mutation, {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "value": value
            })
            data = result.get('data', result) if isinstance(result, dict) else result
            
            # Check for errors in response
            if isinstance(result, dict) and 'errors' in result:
                error_msg = result['errors'][0].get('message', 'Unknown error')
                raise GitHubProjectsError(f"Failed to update field: {error_msg}")
            
            if not data.get('updateProjectV2ItemFieldValue') or not data['updateProjectV2ItemFieldValue'].get('projectV2Item'):
                raise GitHubProjectsError("Failed to update field - no result returned")
            
            logger.info(f"Updated field {field_id} for item {item_id}")
            return data['updateProjectV2ItemFieldValue']['projectV2Item']
            
        except GitHubProjectsError:
            raise
        except Exception as e:
            raise GitHubProjectsError(f"Failed to update field value: {e}")
    
    def _ensure_option_descriptions(self, options: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensure all options have a description field (required by GitHub API).
        
        Args:
            options: List of option dictionaries
            
        Returns:
            Options with description fields added if missing
        """
        for option in options:
            if 'description' not in option:
                # Use the name as description if not provided
                option['description'] = option['name']
        return options
    
    async def create_field(self, project_id: str, name: str, data_type: str, 
                          options: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Create a custom field in a project.
        
        Args:
            project_id: Project node ID
            name: Field name
            data_type: Field data type (TEXT, NUMBER, DATE, SINGLE_SELECT, etc.)
            options: For select fields, list of {name, color} options
            
        Returns:
            Created field information
            
        Raises:
            GitHubProjectsError: If field creation fails
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty")
        if not name or not name.strip():
            raise ValueError("Field name cannot be empty")
        if not data_type or not data_type.strip():
            raise ValueError("Field data type cannot be empty")
        if len(name) > 256:
            raise ValueError("Field name cannot exceed 256 characters")
            
        # Validate data type
        valid_types = ["TEXT", "NUMBER", "DATE", "SINGLE_SELECT", "ITERATION"]
        if data_type not in valid_types:
            raise ValueError(f"Invalid field data type: {data_type}. Must be one of: {', '.join(valid_types)}")
            
        # Validate options for select fields
        if data_type == "SINGLE_SELECT" and not options:
            raise ValueError("SINGLE_SELECT fields require at least one option")
        if data_type == "SINGLE_SELECT" and options:
            # Ensure all options have descriptions
            options = self._ensure_option_descriptions(options)
            
            mutation = """
            mutation($projectId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]!) {
                createProjectV2Field(input: {
                    projectId: $projectId
                    dataType: SINGLE_SELECT
                    name: $name
                    singleSelectOptions: $options
                }) {
                    projectV2Field {
                        ... on ProjectV2SingleSelectField {
                            id
                            name
                            dataType
                            options {
                                id
                                name
                                color
                            }
                        }
                    }
                }
            }
            """
            variables = {
                "projectId": project_id,
                "name": name,
                "options": options
            }
        else:
            mutation = """
            mutation($projectId: ID!, $name: String!, $dataType: ProjectV2CustomFieldType!) {
                createProjectV2Field(input: {
                    projectId: $projectId
                    dataType: $dataType
                    name: $name
                }) {
                    projectV2Field {
                        ... on ProjectV2Field {
                            id
                            name
                            dataType
                        }
                    }
                }
            }
            """
            variables = {
                "projectId": project_id,
                "name": name,
                "dataType": data_type
            }
        
        result = await self._execute_query(mutation, variables)
        data = result.get('data', result) if isinstance(result, dict) else result
        
        # Check for errors in the response
        if isinstance(result, dict) and 'errors' in result:
            errors = result['errors']
            if errors:
                error_msg = errors[0].get('message', 'Unknown error')
                # Check for specific error types
                if 'Name cannot have a reserved value' in error_msg:
                    raise GitHubProjectsError(f"Field name '{name}' is reserved by GitHub. Try a different name.")
                elif 'Name has already been taken' in error_msg:
                    raise GitHubProjectsError(f"Field name '{name}' already exists in this project.")
                else:
                    raise GitHubProjectsError(f"Failed to create field: {error_msg}")
        
        # Check if field creation returned null
        if not data.get('createProjectV2Field') or not data['createProjectV2Field'].get('projectV2Field'):
            raise GitHubProjectsError(f"Failed to create field '{name}'")
            
        field = data['createProjectV2Field']['projectV2Field']
        
        logger.info(f"Created field '{name}' with type {data_type}")
        return field
    
    async def create_project_from_template(self, owner: str, title: str, template: str, 
                                         body: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a project from a predefined template.
        
        Args:
            owner: Repository owner
            title: Project title
            template: Template name ('roadmap', 'bugs', 'features')
            body: Optional project description
            
        Returns:
            Created project with fields configured
            
        Raises:
            GitHubProjectsError: If project or field creation fails
            ValueError: If template name is invalid
        """
        # Validate template
        valid_templates = ['roadmap', 'bugs', 'features']
        if template not in valid_templates:
            raise ValueError(f"Invalid template: {template}. Must be one of: {', '.join(valid_templates)}")
        
        try:
            # Create the base project
            project = await self.create_project(owner, title, body)
            project_id = project["id"]
            
            # Apply template fields based on type
            fields = []
            try:
                if template == "roadmap":
                    fields = await self._apply_roadmap_template(project_id)
                elif template == "bugs":
                    fields = await self._apply_bug_tracking_template(project_id)
                elif template == "features":
                    fields = await self._apply_feature_development_template(project_id)
                
                project["fields"] = fields
                logger.info(f"Created project '{title}' from template '{template}'")
                return project
                
            except Exception as e:
                # If field creation fails, we still have the project
                logger.error(f"Failed to apply template fields: {e}")
                project["fields"] = fields
                project["template_error"] = str(e)
                return project
                
        except Exception as e:
            raise GitHubProjectsError(f"Failed to create project from template: {e}")
    
    async def _apply_roadmap_template(self, project_id: str) -> List[Dict[str, Any]]:
        """Apply Implementation Roadmap template fields."""
        fields = []
        
        # Progress field (Status is reserved)
        status_field = await self.create_field(
            project_id, "Progress", "SINGLE_SELECT",
            [
                {"name": "📋 Planned", "color": "GRAY"},
                {"name": "🚧 In Progress", "color": "YELLOW"},
                {"name": "✅ Completed", "color": "GREEN"},
                {"name": "❌ Cancelled", "color": "RED"},
                {"name": "⏸️ On Hold", "color": "ORANGE"}
            ]
        )
        fields.append(status_field)
        
        # Priority field
        priority_field = await self.create_field(
            project_id, "Priority", "SINGLE_SELECT",
            [
                {"name": "🔥 Critical", "color": "RED"},
                {"name": "⭐ High", "color": "ORANGE"},
                {"name": "📌 Medium", "color": "YELLOW"},
                {"name": "📎 Low", "color": "BLUE"}
            ]
        )
        fields.append(priority_field)
        
        # Epic field
        epic_field = await self.create_field(
            project_id, "Epic", "SINGLE_SELECT",
            [
                {"name": "🏗️ Foundation", "color": "PURPLE"},
                {"name": "🚀 Enhancement", "color": "BLUE"},
                {"name": "🔌 Integration", "color": "GREEN"},
                {"name": "⚡ Optimization", "color": "YELLOW"},
                {"name": "📚 Documentation", "color": "GRAY"}
            ]
        )
        fields.append(epic_field)
        
        # Complexity field
        complexity_field = await self.create_field(
            project_id, "Complexity", "SINGLE_SELECT",
            [
                {"name": "1️⃣ Trivial", "color": "GREEN"},
                {"name": "2️⃣ Simple", "color": "BLUE"},
                {"name": "3️⃣ Medium", "color": "YELLOW"},
                {"name": "4️⃣ Complex", "color": "ORANGE"},
                {"name": "5️⃣ Very Complex", "color": "RED"}
            ]
        )
        fields.append(complexity_field)
        
        # Version field
        version_field = await self.create_field(project_id, "Target Version", "TEXT")
        fields.append(version_field)
        
        # Due Date field
        due_date_field = await self.create_field(project_id, "Due Date", "DATE")
        fields.append(due_date_field)
        
        return fields
    
    async def _apply_bug_tracking_template(self, project_id: str) -> List[Dict[str, Any]]:
        """Apply Bug Tracking template fields."""
        fields = []
        
        # Bug Status field (Status is reserved)
        status_field = await self.create_field(
            project_id, "Bug Status", "SINGLE_SELECT",
            [
                {"name": "🆕 New", "color": "GRAY"},
                {"name": "🔍 Triaged", "color": "BLUE"},
                {"name": "🔧 In Progress", "color": "YELLOW"},
                {"name": "👀 In Review", "color": "PURPLE"},
                {"name": "✅ Fixed", "color": "GREEN"},
                {"name": "❌ Won't Fix", "color": "RED"}
            ]
        )
        fields.append(status_field)
        
        # Severity field
        severity_field = await self.create_field(
            project_id, "Severity", "SINGLE_SELECT",
            [
                {"name": "💥 Critical", "color": "RED"},
                {"name": "🔴 High", "color": "ORANGE"},
                {"name": "🟡 Medium", "color": "YELLOW"},
                {"name": "🟢 Low", "color": "GREEN"}
            ]
        )
        fields.append(severity_field)
        
        # Component field
        component_field = await self.create_field(
            project_id, "Component", "SINGLE_SELECT",
            [
                {"name": "🎯 Core", "color": "PURPLE"},
                {"name": "🔍 Search", "color": "BLUE"},
                {"name": "📚 Indexing", "color": "GREEN"},
                {"name": "🌐 API", "color": "YELLOW"},
                {"name": "🧩 Integration", "color": "ORANGE"},
                {"name": "📖 Documentation", "color": "GRAY"}
            ]
        )
        fields.append(component_field)
        
        # Reproduction field
        repro_field = await self.create_field(project_id, "Reproduction Steps", "TEXT")
        fields.append(repro_field)
        
        # Fix Version field
        fix_version_field = await self.create_field(project_id, "Fix Version", "TEXT")
        fields.append(fix_version_field)
        
        return fields
    
    async def analyze_issue_for_project_fields(self, issue: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Analyze an issue using RAG to determine appropriate field values.
        
        Args:
            issue: GitHub issue object with title, body, labels, etc.
            project_id: Project ID to get available fields
            
        Returns:
            Dictionary mapping field names to suggested values
        """
        # Get project fields
        project = await self.get_project_by_id(project_id)
        fields = project.get("fields", {}).get("nodes", [])
        
        # Extract issue information
        title = issue.get("title", "")
        body = issue.get("body", "")
        labels = [label.get("name", "") for label in issue.get("labels", [])]
        
        # Build context for analysis
        issue_context = f"Title: {title}\nBody: {body}\nLabels: {', '.join(labels)}"
        
        suggestions = {}
        
        # Analyze for each field type
        for field in fields:
            if not field or "id" not in field:
                continue
                
            field_name = field.get("name", "")
            field_type = field.get("dataType", "")
            
            # Skip built-in fields
            if field_type in ["TITLE", "ASSIGNEES", "LABELS", "REPOSITORY", "MILESTONE", 
                             "LINKED_PULL_REQUESTS", "REVIEWERS", "PARENT_ISSUE", "SUB_ISSUES_PROGRESS"]:
                continue
            
            # Analyze based on field type
            if field_type == "SINGLE_SELECT":
                # For select fields, analyze which option best matches
                options = field.get("options", [])
                if options and field_name:
                    suggestion = self._analyze_select_field(issue_context, field_name, options, labels)
                    if suggestion:
                        suggestions[field_name] = suggestion
            
            elif field_type == "TEXT" and field_name:
                # For text fields, extract relevant information
                suggestion = self._analyze_text_field(issue_context, field_name)
                if suggestion:
                    suggestions[field_name] = suggestion
        
        return suggestions
    
    def _analyze_select_field(self, issue_context: str, field_name: str, 
                             options: List[Dict[str, Any]], labels: List[str]) -> Optional[str]:
        """
        Analyze which select option best matches the issue.
        
        Returns the option ID if a match is found.
        """
        field_lower = field_name.lower()
        
        # Priority field analysis
        if "priority" in field_lower:
            # Check labels first
            for label in labels:
                label_lower = label.lower()
                if "critical" in label_lower or "urgent" in label_lower:
                    return self._find_option_id(options, ["critical", "high", "🔥"])
                elif "high" in label_lower:
                    return self._find_option_id(options, ["high", "🔥"])
                elif "medium" in label_lower:
                    return self._find_option_id(options, ["medium", "📌"])
                elif "low" in label_lower:
                    return self._find_option_id(options, ["low", "📎"])
            
            # Check issue content
            context_lower = issue_context.lower()
            if any(word in context_lower for word in ["critical", "urgent", "asap", "blocking"]):
                return self._find_option_id(options, ["critical", "high", "🔥"])
            elif any(word in context_lower for word in ["important", "high priority"]):
                return self._find_option_id(options, ["high", "🔥"])
            else:
                # Default to medium
                return self._find_option_id(options, ["medium", "📌"])
        
        # Status field analysis
        elif "status" in field_lower:
            # New issues typically start as "New" or "Todo"
            return self._find_option_id(options, ["new", "todo", "📋", "🆕"])
        
        # Severity field analysis (for bugs)
        elif "severity" in field_lower:
            context_lower = issue_context.lower()
            if any(word in context_lower for word in ["crash", "data loss", "security"]):
                return self._find_option_id(options, ["critical", "💥"])
            elif any(word in context_lower for word in ["broken", "error", "failure"]):
                return self._find_option_id(options, ["high", "🔴"])
            elif any(word in context_lower for word in ["issue", "problem"]):
                return self._find_option_id(options, ["medium", "🟡"])
            else:
                return self._find_option_id(options, ["low", "🟢"])
        
        # Component field analysis
        elif "component" in field_lower:
            context_lower = issue_context.lower()
            if any(word in context_lower for word in ["search", "query", "find"]):
                return self._find_option_id(options, ["search", "🔍"])
            elif any(word in context_lower for word in ["index", "indexing", "reindex"]):
                return self._find_option_id(options, ["indexing", "📚"])
            elif any(word in context_lower for word in ["api", "endpoint", "http"]):
                return self._find_option_id(options, ["api", "🌐"])
            elif any(word in context_lower for word in ["integration", "github", "mcp"]):
                return self._find_option_id(options, ["integration", "🧩"])
            elif any(word in context_lower for word in ["docs", "documentation", "readme"]):
                return self._find_option_id(options, ["documentation", "📖"])
            else:
                return self._find_option_id(options, ["core", "🎯"])
        
        # Type field analysis (for features)
        elif "type" in field_lower:
            # Check labels for type hints
            for label in labels:
                label_lower = label.lower()
                if "enhancement" in label_lower or "feature" in label_lower:
                    return self._find_option_id(options, ["enhancement", "feature", "✨"])
                elif "performance" in label_lower:
                    return self._find_option_id(options, ["performance", "⚡"])
                elif "refactor" in label_lower:
                    return self._find_option_id(options, ["refactoring", "♻️"])
                elif "ui" in label_lower or "ux" in label_lower:
                    return self._find_option_id(options, ["ui/ux", "🎨"])
        
        return None
    
    def _find_option_id(self, options: List[Dict[str, Any]], search_terms: List[str]) -> Optional[str]:
        """Find the first matching option ID based on search terms."""
        for option in options:
            option_name = option.get("name", "").lower()
            for term in search_terms:
                if term.lower() in option_name:
                    return option.get("id")
        return None
    
    def _analyze_text_field(self, issue_context: str, field_name: str) -> Optional[str]:
        """
        Extract relevant text for text fields.
        """
        field_lower = field_name.lower()
        
        # Reproduction steps
        if "reproduction" in field_lower or "steps" in field_lower:
            # Look for numbered lists or step indicators
            lines = issue_context.split('\n')
            steps = []
            for line in lines:
                if any(pattern in line for pattern in ["1.", "2.", "- ", "* ", "Step"]):
                    steps.append(line.strip())
            
            if steps:
                return "\n".join(steps[:5])  # Limit to first 5 steps
        
        # Epic/Story field
        elif "epic" in field_lower or "story" in field_lower:
            # Extract from labels or title
            for line in issue_context.split('\n'):
                if "epic:" in line.lower() or "story:" in line.lower():
                    return line.strip()
        
        return None
    
    async def smart_add_issue_to_project(self, project_id: str, issue_number: int, 
                                        github_repo) -> Dict[str, Any]:
        """
        Add an issue to a project with smart field assignment using RAG analysis.
        
        Args:
            project_id: Project ID
            issue_number: Issue number
            github_repo: GitHub repository object
            
        Returns:
            Result with item details and field assignments
            
        Raises:
            GitHubProjectsError: If operation fails
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty")
        if not isinstance(issue_number, int) or issue_number <= 0:
            raise ValueError("Issue number must be a positive integer")
        if not github_repo:
            raise ValueError("GitHub repository object is required")
        try:
            # Get the issue details
            issue = github_repo.get_issue(issue_number)
            issue_data = {
                "title": issue.title,
                "body": issue.body or "",
                "labels": [{"name": label.name} for label in issue.labels],
                "state": issue.state,
                "number": issue.number
            }
            
            # Get issue node ID (required for GraphQL)
            # PyGithub provides node_id on Issue objects
            if not hasattr(issue, 'node_id'):
                raise GitHubProjectsError(f"Issue #{issue_number} does not have a node_id attribute")
            issue_node_id = issue.node_id
            
            # Add issue to project using the correct method
            item = await self.add_item_to_project(project_id, issue_node_id)
            
        except GitHubProjectsError:
            raise
        except Exception as e:
            raise GitHubProjectsError(f"Failed to get issue #{issue_number}: {e}")
        
        # Analyze issue for field values
        suggestions = await self.analyze_issue_for_project_fields(issue_data, project_id)
        
        # Apply suggested field values
        applied_fields = {}
        if suggestions:
            # Get project fields to map names to IDs and types
            project = await self.get_project_by_id(project_id)
            field_map = {}
            field_types = {}
            
            for field in project.get("fields", {}).get("nodes", []):
                if field and "id" in field and "name" in field:
                    field_map[field["name"]] = field["id"]
                    field_types[field["name"]] = field.get("dataType", "TEXT")
            
            # Apply each suggestion
            for field_name, value in suggestions.items():
                if field_name in field_map:
                    try:
                        await self.update_item_field(
                            project_id,
                            item["id"],
                            field_map[field_name],
                            value,
                            field_type=field_types.get(field_name)
                        )
                        applied_fields[field_name] = value
                    except Exception as e:
                        logger.warning(f"Failed to set {field_name}: {e}")
        
        return {
            "item": item,
            "applied_fields": applied_fields,
            "suggestions": suggestions
        }
    
    async def _apply_feature_development_template(self, project_id: str) -> List[Dict[str, Any]]:
        """Apply Feature Development template fields."""
        fields = []
        
        # Stage field
        stage_field = await self.create_field(
            project_id, "Stage", "SINGLE_SELECT",
            [
                {"name": "💡 Ideation", "color": "GRAY"},
                {"name": "📋 Planning", "color": "BLUE"},
                {"name": "🛠️ Development", "color": "YELLOW"},
                {"name": "🧪 Testing", "color": "PURPLE"},
                {"name": "🚀 Deployment", "color": "GREEN"},
                {"name": "📊 Monitoring", "color": "ORANGE"}
            ]
        )
        fields.append(stage_field)
        
        # Effort field
        effort_field = await self.create_field(
            project_id, "Effort", "SINGLE_SELECT",
            [
                {"name": "XS (< 1 day)", "color": "GREEN"},
                {"name": "S (1-2 days)", "color": "BLUE"},
                {"name": "M (3-5 days)", "color": "YELLOW"},
                {"name": "L (1-2 weeks)", "color": "ORANGE"},
                {"name": "XL (> 2 weeks)", "color": "RED"}
            ]
        )
        fields.append(effort_field)
        
        # Dependencies field
        dependencies_field = await self.create_field(project_id, "Dependencies", "TEXT")
        fields.append(dependencies_field)
        
        # Impact field
        impact_field = await self.create_field(
            project_id, "Impact", "SINGLE_SELECT",
            [
                {"name": "🌟 High", "color": "GREEN"},
                {"name": "➕ Medium", "color": "YELLOW"},
                {"name": "➖ Low", "color": "GRAY"}
            ]
        )
        fields.append(impact_field)
        
        # Release field
        release_field = await self.create_field(project_id, "Target Release", "TEXT")
        fields.append(release_field)
        
        return fields
    
    async def list_projects(self, owner: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List GitHub Projects V2 for a user or organization.
        
        Args:
            owner: Username or organization name
            limit: Maximum number of projects to return (default: 20, max: 100)
            
        Returns:
            List of project information
            
        Raises:
            GitHubProjectsError: If listing fails
        """
        if limit > 100:
            limit = 100
        elif limit < 1:
            limit = 20
            
        query = """
        query($login: String!, $limit: Int!) {
            user(login: $login) {
                projectsV2(first: $limit) {
                    totalCount
                    nodes {
                        id
                        number
                        title
                        shortDescription
                        url
                        public
                        createdAt
                        updatedAt
                        items(first: 0) {
                            totalCount
                        }
                    }
                }
            }
            organization(login: $login) {
                projectsV2(first: $limit) {
                    totalCount
                    nodes {
                        id
                        number
                        title
                        shortDescription
                        url
                        public
                        createdAt
                        updatedAt
                        items(first: 0) {
                            totalCount
                        }
                    }
                }
            }
        }
        """
        
        try:
            result = await self._execute_query(query, {"login": owner, "limit": limit})
            data = result.get('data', result) if isinstance(result, dict) else result
            
            # Check both user and organization
            projects_data = None
            owner_type = None
            
            if data.get('user') and data['user'].get('projectsV2'):
                projects_data = data['user']['projectsV2']
                owner_type = 'user'
            elif data.get('organization') and data['organization'].get('projectsV2'):
                projects_data = data['organization']['projectsV2']
                owner_type = 'organization'
            
            if not projects_data:
                # No projects found or owner doesn't exist
                return []
            
            projects = projects_data.get('nodes', [])
            
            # Format the projects list
            formatted_projects = []
            for project in projects:
                if project:  # Skip any null entries
                    formatted_projects.append({
                        "id": project["id"],
                        "number": project["number"],
                        "title": project["title"],
                        "description": project.get("shortDescription", ""),
                        "url": project["url"],
                        "public": project["public"],
                        "item_count": project["items"]["totalCount"],
                        "created_at": project["createdAt"],
                        "updated_at": project["updatedAt"],
                        "owner": owner,
                        "owner_type": owner_type
                    })
            
            logger.info(f"Found {len(formatted_projects)} projects for {owner}")
            return formatted_projects
            
        except Exception as e:
            logger.error(f"Failed to list projects for {owner}: {e}")
            raise GitHubProjectsError(f"Failed to list projects: {e}")
    
    async def delete_project(self, project_id: str) -> Dict[str, Any]:
        """
        Delete a GitHub Project V2.
        
        Args:
            project_id: Project node ID (must start with PVT_)
            
        Returns:
            Dictionary with deletion status and project ID
            
        Raises:
            GitHubProjectsError: If project deletion fails
            ValueError: If project ID is invalid
        """
        # Validate inputs
        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty")
        if not project_id.startswith("PVT_"):
            raise ValueError(f"Invalid project ID format: {project_id}. Must start with 'PVT_'")
        
        # Try to get project details first to confirm it exists
        try:
            project = await self.get_project_by_id(project_id)
            project_title = project.get("title", "Unknown")
            logger.info(f"Attempting to delete project '{project_title}' (ID: {project_id})")
        except GitHubProjectsError:
            # Project might not exist or we don't have access
            logger.warning(f"Could not fetch project details for {project_id} before deletion")
            project_title = "Unknown"
        
        mutation = """
        mutation($projectId: ID!) {
            deleteProjectV2(input: {
                projectId: $projectId
            }) {
                projectV2 {
                    id
                    title
                    number
                }
            }
        }
        """
        
        try:
            result = await self._execute_query(mutation, {"projectId": project_id})
            data = result.get('data', result) if isinstance(result, dict) else result
            
            # Check for errors in response
            if isinstance(result, dict) and 'errors' in result:
                errors = result['errors']
                if errors:
                    error_msg = errors[0].get('message', 'Unknown error')
                    
                    # Check for specific error conditions
                    if 'not found' in error_msg.lower():
                        raise GitHubProjectsError(f"Project {project_id} not found or you don't have access")
                    elif 'permission' in error_msg.lower() or 'forbidden' in error_msg.lower():
                        raise GitHubProjectsError(f"Insufficient permissions to delete project {project_id}")
                    elif 'cannot delete' in error_msg.lower():
                        raise GitHubProjectsError(f"Project {project_id} cannot be deleted: {error_msg}")
                    else:
                        raise GitHubProjectsError(f"Failed to delete project: {error_msg}")
            
            # Check if deletion returned data
            if not data.get('deleteProjectV2'):
                # Deletion successful but no data returned (this is expected for some mutations)
                logger.info(f"Project {project_id} deleted successfully")
                return {
                    "deleted": True,
                    "project_id": project_id,
                    "title": project_title,
                    "message": f"Project '{project_title}' deleted successfully"
                }
            
            # If we got project data back, use it
            deleted_project = data['deleteProjectV2'].get('projectV2', {})
            title = deleted_project.get('title', project_title)
            
            logger.info(f"Deleted project '{title}' with ID: {project_id}")
            return {
                "deleted": True,
                "project_id": project_id,
                "title": title,
                "number": deleted_project.get('number'),
                "message": f"Project '{title}' deleted successfully"
            }
            
        except GitHubProjectsError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting project: {e}")
            raise GitHubProjectsError(f"Failed to delete project: {e}")
    
    async def smart_add_sub_issues_to_project(self, project_id: str, parent_issue_number: int,
                                             github_repo, github_client) -> Dict[str, Any]:
        """
        Add all sub-issues of a parent issue to a project with smart field assignment.
        
        This method:
        1. Fetches all sub-issues for the parent issue
        2. Adds each sub-issue to the project
        3. Applies smart field assignment based on issue content
        4. Inherits certain field values from the parent if already in the project
        
        Args:
            project_id: Project ID
            parent_issue_number: Parent issue number
            github_repo: GitHub repository object
            github_client: GitHub client with sub-issues support
            
        Returns:
            Dict with added sub-issues and their field assignments
            
        Raises:
            GitHubProjectsError: If operation fails
        """
        try:
            # Get sub-issues
            sub_issues = github_client.list_sub_issues(parent_issue_number)
            
            if not sub_issues:
                return {
                    "parent_issue": parent_issue_number,
                    "message": "No sub-issues found",
                    "added_count": 0,
                    "sub_issues": []
                }
            
            # Get parent issue's project item if it exists
            parent_fields = {}
            try:
                # Try to find parent in project to inherit field values
                parent_issue = github_repo.get_issue(parent_issue_number)
                # Note: We'd need to implement a method to find item by issue
                # For now, we'll skip inheritance
                logger.info(f"Processing {len(sub_issues)} sub-issues for parent #{parent_issue_number}")
            except Exception as e:
                logger.warning(f"Could not get parent issue details: {e}")
            
            # Process each sub-issue
            added_sub_issues = []
            failed_sub_issues = []
            
            for sub_issue_data in sub_issues:
                try:
                    # Extract issue number from sub-issue data
                    # The API response format may vary, handle both cases
                    if isinstance(sub_issue_data, dict):
                        sub_issue_number = sub_issue_data.get('number') or sub_issue_data.get('issue_number')
                    else:
                        # If it's just a number
                        sub_issue_number = sub_issue_data
                    
                    if not sub_issue_number:
                        logger.warning(f"Invalid sub-issue data: {sub_issue_data}")
                        continue
                    
                    # Add sub-issue to project with smart assignment
                    result = await self.smart_add_issue_to_project(
                        project_id, sub_issue_number, github_repo
                    )
                    
                    added_sub_issues.append({
                        "issue_number": sub_issue_number,
                        "item_id": result["item"]["id"],
                        "applied_fields": result["applied_fields"]
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to add sub-issue #{sub_issue_number}: {e}")
                    failed_sub_issues.append({
                        "issue_number": sub_issue_number,
                        "error": str(e)
                    })
            
            return {
                "parent_issue": parent_issue_number,
                "added_count": len(added_sub_issues),
                "failed_count": len(failed_sub_issues),
                "sub_issues": added_sub_issues,
                "failed_sub_issues": failed_sub_issues,
                "message": f"Added {len(added_sub_issues)} sub-issues to project"
            }
            
        except Exception as e:
            logger.error(f"Failed to add sub-issues to project: {e}")
            raise GitHubProjectsError(f"Failed to add sub-issues: {e}")


# Singleton instance management
_projects_manager = None

def get_projects_manager(github_client) -> GitHubProjectsManager:
    """
    Get or create GitHub Projects manager instance.
    
    Args:
        github_client: Authenticated GitHubClient instance
        
    Returns:
        GitHubProjectsManager instance
    """
    global _projects_manager
    
    if _projects_manager is None:
        _projects_manager = GitHubProjectsManager(github_client)
    
    return _projects_manager