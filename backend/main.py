"""FastAPI backend server for Advanced RAG Demo."""
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.config import FEEDBACK_FILE, PDF_DIR
from backend.models import (
    QueryRequest, QueryResponse, FeedbackRequest, FeedbackResponse,
    StatsResponse, HealthResponse, ErrorResponse, EvaluationResponse, ConfigRequest
)
from backend.feedback_system import ChatbotFeedbackSystem
from backend.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
feedback_system = None
rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    global feedback_system, rag_pipeline
    
    logger.info("Starting RAG Backend...")
    
    from backend.config import PROVIDER
    logger.info(f"Using {PROVIDER}")
    
    feedback_system = ChatbotFeedbackSystem(save_path=str(FEEDBACK_FILE))
    logger.info(f"Feedback system loaded: {len(feedback_system.feedback_history)} entries")
    
    rag_pipeline = RAGPipeline(feedback_system)
    
    # Load PDFs if they exist
    pdf_count = rag_pipeline.load_pdf_data()
    if pdf_count > 0:
        logger.info(f"Loaded {pdf_count} PDF documents")
        try:
            chunk_count = rag_pipeline.build_vectorstore()
            logger.info(f"Vectorstore ready: {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Failed to build vectorstore: {e}")
    else:
        logger.info("No PDFs found. Upload PDFs to start using the system.")
    
    # COMMENTED OUT: SQuAD dataset loading - system now uses PDFs only
    # logger.info("Loading SQuAD dataset...")
    # try:
    #     doc_count = rag_pipeline.load_squad_data(max_examples=2000)
    #     logger.info(f"Loaded {doc_count} documents")
    # except Exception as e:
    #     logger.error(f"Failed to load dataset: {e}")
    
    logger.info("Backend ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down backend")


app = FastAPI(
    title="Advanced RAG Demo API",
    description="Backend API for RAG chatbot with adaptive feedback learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced RAG Demo API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from backend.config import PROVIDER
    return HealthResponse(
        status="healthy",
        provider=PROVIDER,
        vectorstore_loaded=rag_pipeline.vectordb is not None,
        feedback_system_loaded=feedback_system is not None,
        total_documents=len(rag_pipeline.base_docs),
        total_chunks=len(rag_pipeline.chunks)
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a user query through the RAG pipeline."""
    try:
        logger.info(f"Query received: {request.query[:50]}...")
        
        # Lazy load vectorstore if not initialized
        if rag_pipeline.vectordb is None:
            logger.info("Vectorstore not initialized, loading now...")
            if not rag_pipeline.base_docs:
                logger.info("Loading documents...")
                rag_pipeline.load_squad_data(max_examples=600)
            logger.info("Building/loading vectorstore...")
            rag_pipeline.build_vectorstore()
            logger.info("Vectorstore ready")
        
        result = rag_pipeline.query(
            request.query,
            request.temperature
        )
        
        logger.info(f"Response generated ({len(result['response'].split())} words)")
        logger.info(f"Rewritten queries: {len(result['rewritten_queries'])}")
        logger.info(f"Stage 1 candidates: {result['stage1_candidates']}, Stage 2 reranked: {result['stage2_reranked']}")
        
        return QueryResponse(
            query=request.query,
            rewritten_queries=result['rewritten_queries'],
            response=result['response'],
            response_length=len(result['response'].split()),
            retrieved_docs=result['stage2_reranked'],
            stage1_candidates=result['stage1_candidates'],
            stage2_reranked=result['stage2_reranked'],
            retrieval_scores=result['retrieval_scores'],
            chunks_used=result['chunks_used'],
            evaluation=result['evaluation'],
            memory_context=result['memory_context'],
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a response."""
    try:
        logger.info(f"Feedback received: {request.rating}")
        
        feedback_system.add_feedback(
            query=request.query,
            response=request.response,
            rating=request.rating,
            comment=request.comment,
            context_used=request.context or ""
        )
        
        insights = feedback_system.get_improvement_insights()
        
        return FeedbackResponse(
            message="Feedback recorded successfully",
            total_feedback=insights["total_feedback"],
            satisfaction_rate=insights["satisfaction_rate"],
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get feedback statistics and system status."""
    try:
        insights = feedback_system.get_improvement_insights()
        trend = feedback_system.get_recent_improvement_trend()
        system_prompt = feedback_system.generate_system_prompt()
        
        return StatsResponse(
            total_feedback=insights["total_feedback"],
            positive_count=insights["positive_count"],
            negative_count=insights["negative_count"],
            satisfaction_rate=insights["satisfaction_rate"],
            positive_avg_length=insights["positive_avg_length"],
            negative_avg_length=insights["negative_avg_length"],
            top_issues=insights["top_issues"],
            recent_trend=trend,
            system_prompt=system_prompt
        )
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation", response_model=EvaluationResponse)
async def get_evaluation():
    """Get evaluation metrics for RAG pipeline performance."""
    try:
        eval_summary = rag_pipeline.get_evaluation_summary()
        
        return EvaluationResponse(
            average_retrieval_quality=eval_summary["average_retrieval_quality"],
            average_response_relevance=eval_summary["average_response_relevance"],
            average_response_completeness=eval_summary["average_response_completeness"],
            total_queries_evaluated=eval_summary["total_queries_evaluated"],
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error getting evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf")
async def upload_pdf(files: list[UploadFile] = File(...)):
    """Upload and process multiple PDF documents with verbose feedback."""
    results = []
    
    try:
        logger.info(f"Received {len(files)} file(s) for upload")
        
        for idx, file in enumerate(files, 1):
            if not file.filename.endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {file.filename}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Only PDF files allowed"
                })
                continue
            
            logger.info(f"[{idx}/{len(files)}] Processing {file.filename}...")
            
            # Save to PDF directory with original filename
            pdf_path = PDF_DIR / file.filename
            with open(pdf_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            # Load PDF
            logger.info(f"[{idx}/{len(files)}] Loading pages from {file.filename}...")
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            
            # Override metadata to use clean filename
            for doc in docs:
                doc.metadata["source"] = file.filename
            
            # Add to pipeline
            rag_pipeline.base_docs.extend(docs)
            
            logger.info(f"[{idx}/{len(files)}] Loaded {len(docs)} pages from {file.filename}")
            
            results.append({
                "filename": file.filename,
                "success": True,
                "pages": len(docs)
            })
        
        # Rebuild vectorstore with all PDFs
        if any(r["success"] for r in results):
            logger.info("Chunking documents...")
            logger.info(f"Total documents to process: {len(rag_pipeline.base_docs)}")
            
            chunk_count = rag_pipeline.build_vectorstore(force_rebuild=True)
            
            logger.info(f"✓ Processing complete: {chunk_count} chunks created")
            
            return {
                "success": True,
                "files": results,
                "total_chunks": chunk_count,
                "total_documents": len(rag_pipeline.base_docs)
            }
        else:
            raise HTTPException(status_code=400, detail="No valid PDF files processed")
    
    except Exception as e:
        logger.error(f"Error uploading PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_config")
async def update_config(config: ConfigRequest):
    """Update RAG pipeline parameters."""
    try:
        from backend import config as backend_config
        
        # Track if we need to rebuild vectorstore
        needs_rebuild = False
        
        if config.chunk_size:
            backend_config.CHUNK_SIZE = config.chunk_size
            needs_rebuild = True
        if config.chunk_overlap:
            backend_config.CHUNK_OVERLAP = config.chunk_overlap
            needs_rebuild = True
        if config.stage1_k:
            backend_config.STAGE1_K = config.stage1_k
        if config.top_k:
            backend_config.TOP_K_RERANKED = config.top_k
        
        logger.info(f"Config updated: {config.dict(exclude_none=True)}")
        
        # Rebuild vectorstore if chunk parameters changed
        if needs_rebuild and rag_pipeline.base_docs:
            logger.info("Rebuilding vectorstore with new chunk parameters...")
            chunk_count = rag_pipeline.build_vectorstore(force_rebuild=True)
            logger.info(f"Vectorstore rebuilt: {chunk_count} chunks")
            return {
                "success": True, 
                "message": f"Configuration updated and vectorstore rebuilt with {chunk_count} chunks"
            }
        
        return {"success": True, "message": "Configuration updated"}
    
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks")
async def get_chunks():
    """Get all chunks for viewing."""
    try:
        if not rag_pipeline.chunks:
            return {"chunks": [], "total": 0}
        
        chunks_data = []
        for i, chunk in enumerate(rag_pipeline.chunks):
            chunks_data.append({
                "id": i + 1,
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "Unknown"),
                "length": len(chunk.page_content)
            })
        
        return {"chunks": chunks_data, "total": len(chunks_data)}
    
    except Exception as e:
        logger.error(f"Error getting chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT
    
    # Set reload=False to avoid double startup
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False
    )
