"""API client for backend communication."""
import requests
from typing import Dict, Any, Optional


class APIClient:
    """Client for interacting with the backend API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def query(self, query: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Send a query to the RAG pipeline."""
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": query, "temperature": temperature}
        )
        response.raise_for_status()
        return response.json()
    
    def submit_feedback(
        self,
        query: str,
        response: str,
        rating: str,
        comment: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit user feedback."""
        payload = {
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment,
            "context": context
        }
        response = requests.post(f"{self.base_url}/feedback", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def get_evaluation(self) -> Dict[str, Any]:
        """Get evaluation metrics."""
        response = requests.get(f"{self.base_url}/evaluation")
        response.raise_for_status()
        return response.json()
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update RAG configuration parameters."""
        response = requests.post(f"{self.base_url}/update_config", json=config)
        response.raise_for_status()
        return response.json()
    
    def get_chunks(self) -> Dict[str, Any]:
        """Get all chunks for viewing."""
        response = requests.get(f"{self.base_url}/chunks")
        response.raise_for_status()
        return response.json()
