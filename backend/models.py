"""Pydantic models for API requests and responses."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User's question", min_length=1)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")


class ConfigRequest(BaseModel):
    """Request model for updating RAG parameters."""
    chunk_size: Optional[int] = Field(None, ge=100, le=1000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=200)
    stage1_k: Optional[int] = Field(None, ge=5, le=100)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    rewritten_queries: List[str] = Field(default_factory=list, description="Rewritten query variations")
    response: str
    response_length: int
    retrieved_docs: int
    stage1_candidates: int = Field(0, description="Initial retrieval count")
    stage2_reranked: int = Field(0, description="Final reranked count")
    retrieval_scores: List[float] = Field(default_factory=list, description="Reranking scores")
    chunks_used: List[str] = Field(default_factory=list, description="Actual text chunks used")
    evaluation: Dict[str, Any] = Field(default_factory=dict, description="Answer quality metrics")
    memory_context: Optional[str] = Field(None, description="Conversation memory used")
    success: bool = True


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    rating: str = Field(..., description="User rating: 👍 or 👎")
    comment: Optional[str] = Field(None, description="Optional feedback comment")
    context: Optional[str] = Field(None, description="Context used for generation")


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    message: str
    total_feedback: int
    satisfaction_rate: float
    success: bool = True


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    total_feedback: int
    positive_count: int
    negative_count: int
    satisfaction_rate: float
    positive_avg_length: int
    negative_avg_length: int
    top_issues: List[tuple]
    recent_trend: Dict[str, Any]
    system_prompt: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    provider: str
    vectorstore_loaded: bool
    feedback_system_loaded: bool
    total_documents: int
    total_chunks: int


class EvaluationResponse(BaseModel):
    """Response model for evaluation metrics."""
    average_retrieval_quality: float
    average_response_relevance: float
    average_response_completeness: float
    total_queries_evaluated: int
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
