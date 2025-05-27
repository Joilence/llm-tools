from typing import Union, List, Optional
import ltls
import pytest
from pydantic import BaseModel

"""
Integration test toolkit, include:
- simple tool with one parameter
- simple tool with multiple parameters
- complicated tool with pydantic basemodel for parameters
"""

class SearchQueryParams(BaseModel):
    """Parameters for search query."""
    query: str
    max_results: int = 10
    case_sensitive: bool = False
    categories: Optional[List[str]] = None

class TestIntegrationToolkit(ltls.Toolkit):
    """A test toolkit for integration testing."""

    @ltls.tool_def(name="echo", description="Echo back the input")
    def echo(self, message: str) -> str:
        """Echo back the input message."""
        return f"You said: {message}"

    @ltls.tool_def(name="add", description="Add two numbers")
    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers and return the result."""
        return a + b
    
    @ltls.tool_def(name="search", description="Search for items matching the query")
    def search(self, params: SearchQueryParams) -> List[str]:
        """Search for items matching the given query parameters.
        
        Args:
            params: The search parameters including query string, max results,
                   case sensitivity, and optional categories to filter by.
                   
        Returns:
            A list of matching items.
        """
        # Mock data for the search
        all_items = [
            "Apple", "Banana", "Cherry", 
            "Document1", "Document2", "Document3",
            "User1", "User2", "User3"
        ]
        
        # Filter based on query
        if params.case_sensitive:
            results = [item for item in all_items if params.query in item]
        else:
            results = [item for item in all_items if params.query.lower() in item.lower()]
            
        # Filter by categories if provided
        if params.categories:
            # Mock category filtering (in real implementation, items would have categories)
            if "fruits" in params.categories:
                results = [r for r in results if r in ["Apple", "Banana", "Cherry"]]
            elif "documents" in params.categories:
                results = [r for r in results if r in ["Document1", "Document2", "Document3"]]
            elif "users" in params.categories:
                results = [r for r in results if r in ["User1", "User2", "User3"]]
                
        # Limit results
        return results[:params.max_results]


@pytest.fixture
def toolkit():
    """Fixture to provide a TestIntegrationToolkit instance."""
    return TestIntegrationToolkit()