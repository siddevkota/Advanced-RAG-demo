# PDF-Based RAG Chatbot

A production-ready RAG chatbot that works exclusively with your uploaded PDFs. Features query rewriting, semantic search, reranking, and conversation memory.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create `.env` file:
```
OPENAI_API_KEY=your-openai-key-here
```

### 3. Run

Terminal 1 - Backend:
```bash
python -m uvicorn backend.main:app
```

Terminal 2 - Frontend:
```bash
streamlit run streamlit/app.py
```

Access at: http://localhost:8501

### 4. Upload PDFs

1. Open the Streamlit interface
2. Use the sidebar to upload one or multiple PDFs
3. Click "Process PDFs" to build the knowledge base
4. Start chatting!

## Features

- **PDF-Only System**: Upload your own documents, no pre-loaded datasets
- **Multiple PDF Upload**: Process multiple documents at once
- **Query Rewriting**: Generates 3-4 context-aware query variations
- **Two-Stage Retrieval**: Vector search (OpenAI embeddings) + cross-encoder reranking
- **Conversation Memory**: Understands follow-up questions and pronouns
- **Configurable Parameters**: Adjust chunk size, retrieval count, etc.
- **Quality Metrics**: Auto-evaluation of every response
- **Feedback Learning**: System adapts based on user ratings
- **Strict Guardrails**: Only answers based on uploaded content

## Configuration

All parameters adjustable in sidebar:
- Chunk Size (100-1000)
- Chunk Overlap (0-200)
- Initial Retrieval Count (5-100)
- Final Chunks Used (1-20)
- LLM Temperature (0.0-2.0)

## Architecture

See `ARCHITECTURE.md` for system design details.

## API Endpoints

- `POST /query` - Submit query
- `POST /upload_pdf` - Upload multiple PDF documents
- `POST /feedback` - Submit user feedback  
- `POST /update_config` - Update RAG parameters (rebuilds vectorstore if needed)
- `GET /stats` - Get feedback statistics
- `GET /evaluation` - Get evaluation metrics
- `GET /health` - Health check
- `GET /chunks` - View all chunks (for debugging)

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server
│   ├── rag_pipeline.py      # RAG implementation
│   ├── feedback_system.py   # Adaptive learning
│   ├── models.py            # API schemas
│   └── config.py            # Configuration
├── streamlit/
│   ├── app.py               # Main UI
│   ├── api_client.py        # Backend client
│   └── app_config.py        # UI config
├── data/
│   ├── pdfs/                # Uploaded PDFs
│   ├── vectorstore/         # FAISS index
│   └── chatbot_feedback.json
└── notebooks/               # Jupyter experiments
```

## License

Educational purposes
