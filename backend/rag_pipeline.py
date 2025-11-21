"""RAG pipeline with document retrieval and reranking."""
import torch
import numpy as np
from typing import List, Tuple, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, STAGE1_K, TOP_K_RERANKED, PDF_DIR, VECTORSTORE_PATH,
    PROVIDER, DATA_DIR
)
from backend.feedback_system import ChatbotFeedbackSystem


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""
    
    def __init__(self, feedback_system: ChatbotFeedbackSystem):
        self.feedback_system = feedback_system
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize reranker
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL
        ).to(self.device)
        
        # Initialize vectorstore
        self.vectordb = None
        self.chunks = []
        self.base_docs = []
        
        # Conversation memory
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges
        
        # Evaluation tracking
        self.query_evaluations = []
    
    # def load_squad_data(self, max_examples: int = 600):
    #     """Load SQuAD dataset."""
    #     ds = load_dataset("squad", split="train[:10%]")
    #     ds = ds.shuffle(seed=42).select(range(min(max_examples, len(ds))))
        
    #     contexts = []
    #     for ex in ds:
    #         contexts.append(ex["context"])
        
    #     unique_contexts = list({c: True for c in contexts}.keys())
    #     self.base_docs = [
    #         Document(page_content=c, metadata={"source": f"squad_{i}"})
    #         for i, c in enumerate(unique_contexts)
    #     ]
        
    #     return len(self.base_docs)
    
    def load_pdf_data(self):
        """Load PDF documents."""
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            return 0
        
        self.base_docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            self.base_docs.extend(loader.load())
        
        return len(self.base_docs)
    
    def _get_embeddings(self):
        """Get embeddings model - using OpenAI."""
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    def build_vectorstore(self, force_rebuild: bool = False):
        """Build FAISS vectorstore with chunking."""
        if not self.base_docs:
            raise ValueError("No documents loaded. Call load_squad_data() or load_pdf_data() first.")
        
        # Try to load existing vectorstore first
        vectorstore_file = VECTORSTORE_PATH / "index.faiss"
        if not force_rebuild and vectorstore_file.exists():
            try:
                import pickle
                embeddings = self._get_embeddings()
                self.vectordb = FAISS.load_local(
                    str(VECTORSTORE_PATH),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                # Load chunks info
                chunks_file = VECTORSTORE_PATH / "chunks.pkl"
                if chunks_file.exists():
                    with open(chunks_file, 'rb') as f:
                        self.chunks = pickle.load(f)
                
                # Check if doc count matches - if not, rebuild
                if len(self.base_docs) != len(self.chunks):
                    print(f"⚠️  Document count changed ({len(self.base_docs)} docs vs {len(self.chunks)} chunks)")
                    print(f"   Rebuilding vectorstore...")
                    force_rebuild = True
                else:
                    print(f"✓ Loaded existing vectorstore with {len(self.chunks)} chunks")
                    return len(self.chunks)
            except Exception as e:
                print(f"Warning: Could not load existing vectorstore: {e}")
                force_rebuild = True
        
        # Build new vectorstore
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(self.base_docs)
        
        embeddings = self._get_embeddings()
        print(f"Building vectorstore with {PROVIDER} embeddings...")
        self.vectordb = FAISS.from_documents(self.chunks, embedding=embeddings)
        
        # Save vectorstore for future use
        try:
            import pickle
            import json
            self.vectordb.save_local(str(VECTORSTORE_PATH))
            # Save chunks info
            chunks_file = VECTORSTORE_PATH / "chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Export chunks to JSON for easy viewing
            chunks_json = []
            for i, chunk in enumerate(self.chunks):
                chunks_json.append({
                    "id": i + 1,
                    "content": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown"),
                    "length": len(chunk.page_content)
                })
            
            json_file = DATA_DIR / "chunks_export.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_json, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved vectorstore to {VECTORSTORE_PATH}")
            print(f"✓ Exported {len(chunks_json)} chunks to {json_file}")
        except Exception as e:
            print(f"Warning: Could not save vectorstore: {e}")
        
        return len(self.chunks)
    
    def rewrite_query(self, original_query: str) -> List[str]:
        """Generate query variations for better retrieval with conversation awareness."""
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        
        # Build recent conversation context for query rewriting
        context_info = ""
        if self.conversation_history:
            recent = self.conversation_history[-2:]  # Last 2 exchanges
            context_info = "\n\nRecent conversation:\n" + "\n".join([
                f"User: {h['query']}\nAssistant: {h['response'][:100]}..."
                for h in recent
            ])
        
        # Get recent feedback insights
        feedback_context = self.feedback_system.get_recent_feedback_context()
        
        prompt = f"""Rephrase this question in 3 different ways for better search results.

IMPORTANT RULES:
1. If the question uses pronouns (it, he, she, they, that, this) or is a follow-up, use the conversation context to make the question more specific and standalone
2. For very short questions (like "how?", "why?", "when?"), look at the MOST RECENT exchange to understand what the user is asking about
3. Keep the SAME TOPIC as the previous question when expanding short follow-ups
4. Focus on finding factual information, not abstract relationships

{context_info}

{feedback_context}

Current Question: {original_query}

Rephrase into 3 standalone, specific questions that focus on the SAME TOPIC as recent conversation:
1.
2.
3."""
        
        try:
            response = llm.invoke(prompt).content.strip()
            lines = [l.strip() for l in response.split('\n') if l.strip() and l.strip()[0].isdigit()]
            variations = [original_query] + [l.split('.', 1)[1].strip() if '.' in l else l for l in lines[:3]]
            return variations[:4]
        except Exception as e:
            print(f"Query rewriting error: {e}")
            return [original_query]
    
    def cross_encoder_rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = TOP_K_RERANKED
    ) -> Tuple[List[Document], List[float]]:
        """Rerank documents using cross-encoder."""
        if not docs:
            return [], []
        
        pairs = [(query, d.page_content) for d in docs]
        inputs = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.squeeze(-1).cpu().numpy()
        
        ranked_idx = np.argsort(-scores)
        top_docs = [docs[i] for i in ranked_idx[:top_k]]
        top_scores = [float(scores[i]) for i in ranked_idx[:top_k]]
        
        return top_docs, top_scores
    
    def retrieve_documents(self, query: str, rewritten_queries: List[str] = None) -> Tuple[List[Document], int, List[float]]:
        """Retrieve and rerank documents with detailed tracking."""
        if self.vectordb is None:
            raise ValueError("Vectorstore not initialized. Call build_vectorstore() first.")
        
        # Use all query variations for retrieval
        all_candidates = []
        queries_to_use = rewritten_queries if rewritten_queries else [query]
        
        # Also add context from recent conversation for better retrieval
        if self.conversation_history and len(query.split()) <= 3:  # Short queries like "how?", "why?", "tell me more"
            recent = self.conversation_history[-1]
            # Extract key entities from recent conversation
            context_query = f"{recent['query']}"  # Use previous user question
            if context_query not in queries_to_use:
                queries_to_use.append(context_query)
        
        for q in queries_to_use:
            candidates = self.vectordb.similarity_search(q, k=STAGE1_K)
            all_candidates.extend(candidates)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for doc in all_candidates:
            doc_id = doc.page_content[:100]  # Use first 100 chars as identifier
            if doc_id not in seen:
                seen.add(doc_id)
                unique_candidates.append(doc)
        
        stage1_count = len(unique_candidates)
        top_docs, scores = self.cross_encoder_rerank(query, unique_candidates)
        return top_docs, stage1_count, scores
    
    def generate_response(
        self,
        query: str,
        context_docs: List[Document],
        temperature: float = 0.7
    ) -> Tuple[str, str, str, List[Dict]]:
        """Generate response using LLM with adaptive prompting and memory."""
        # Build numbered context with sources - group by PDF file
        from pathlib import Path
        
        # Group chunks by source PDF
        source_map = {}  # PDF name -> list of chunks
        for doc in context_docs:
            source_file = Path(doc.metadata.get("source", "Unknown")).name
            if source_file not in source_map:
                source_map[source_file] = []
            source_map[source_file].append(doc)
        
        # Build context with PDF-level source numbers
        context_parts = []
        sources = []
        source_id = 1
        
        for pdf_name, docs in source_map.items():
            # Add all chunks from this PDF under one source number
            pdf_content = "\n\n".join([doc.page_content for doc in docs])
            context_parts.append(f"[Source {source_id}: {pdf_name}]\n{pdf_content}")
            
            # Track source info
            pages = [doc.metadata.get("page", "N/A") for doc in docs]
            page_range = f"{min(pages)}-{max(pages)}" if len(set(pages)) > 1 else str(pages[0])
            
            source_info = {
                "id": source_id,
                "source": pdf_name,
                "page": page_range,
                "content": docs[0].page_content[:150] + "...",
                "chunk_count": len(docs)
            }
            sources.append(source_info)
            source_id += 1
        
        context = "\n\n".join(context_parts)
        system_prompt = self.feedback_system.generate_system_prompt()
        
        # Build conversation memory context
        memory_context = ""
        if self.conversation_history:
            memory_context = "\n".join([
                f"User: {h['query']}\nAssistant: {h['response'][:100]}..."
                for h in self.conversation_history[-self.max_history:]
            ])
        
        # Use OpenAI for response generation
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )
        
        # Build memory section
        memory_section = f'Previous Conversation:\n{memory_context}\n\n' if memory_context else ''
        
        # Get feedback guidance for response generation
        feedback_guidance = self.feedback_system.get_recent_feedback_context()
        
        # Strict context guardrail system prompt with memory awareness
        guardrail_prompt = f"""You are a helpful assistant that ONLY answers questions based on the provided context.

STRICT RULES:
1. Use BOTH the provided context AND the previous conversation to answer questions
2. For follow-up questions (like "who was he?", "tell me more", "what about that?"), refer to the previous conversation to understand what the user is asking about
3. If the question cannot be answered using EITHER the context OR the previous conversation, respond EXACTLY with: "I cannot answer this question based on the available context."
4. Do NOT use external knowledge beyond the context and conversation history
5. Do NOT answer questions about current events, personal opinions, or topics not in the context
6. Always base your answer on the context and conversation history provided
7. IMPORTANT: When citing sources, use the format [Source X: filename.pdf] where X is the source number and filename is the actual PDF name
8. For example: "James Bond was an intelligence officer [Source 1: James_Bond.pdf]"
9. Always include the PDF filename when citing sources to make it clear which document the information came from
10. Only cite sources that you actually used in your answer

{feedback_guidance}"""
        
        prompt = f"""{memory_section}Context:
\"\"\"{context}\"\"\"

Question: {query}

Answer:"""
        
        messages = [
            {"role": "system", "content": guardrail_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = llm.invoke(messages).content.strip()
        
        # Store in conversation memory
        self.conversation_history.append({
            "query": query,
            "response": response
        })
        
        return response, context, memory_context, sources
    
    def query(self, query: str, temperature: float = 0.7) -> dict:
        """Complete RAG query pipeline with detailed tracking."""
        # Step 1: Query rewriting
        rewritten_queries = self.rewrite_query(query)
        
        # Step 2: Retrieve and rerank
        docs, stage1_count, scores = self.retrieve_documents(query, rewritten_queries)
        
        # Step 3: Generate response
        response, context, memory_context, sources = self.generate_response(query, docs, temperature)
        
        # Step 4: Evaluate
        evaluation = self.evaluate_response(query, response, context)
        
        # Extract chunks text
        chunks_used = [doc.page_content[:200] + "..." for doc in docs]
        
        return {
            "response": response,
            "context": context,
            "rewritten_queries": rewritten_queries,
            "stage1_candidates": stage1_count,
            "stage2_reranked": len(docs),
            "retrieval_scores": scores,
            "chunks_used": chunks_used,
            "sources": sources,
            "evaluation": evaluation,
            "memory_context": memory_context if memory_context else None
        }
    
    def evaluate_response(self, query: str, response: str, context: str) -> dict:
        """Evaluate response quality."""
        # Simple heuristic-based evaluation
        eval_metrics = {
            "retrieval_quality": 0.0,
            "response_relevance": 0.0,
            "response_completeness": 0.0
        }
        
        # Retrieval quality: Check if key terms from query appear in context
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        overlap = len(query_terms & context_terms) / max(len(query_terms), 1)
        eval_metrics["retrieval_quality"] = min(overlap * 100, 100)
        
        # Response relevance: Check if response uses context
        response_lower = response.lower()
        context_chunks = context.split('\n\n')
        used_chunks = sum(1 for chunk in context_chunks if any(word in response_lower for word in chunk.lower().split()[:5]))
        eval_metrics["response_relevance"] = (used_chunks / max(len(context_chunks), 1)) * 100
        
        # Response completeness: Length and structure
        word_count = len(response.split())
        has_structure = any(char in response for char in ['.', '!', '?'])
        completeness = min((word_count / 30) * 100, 80) + (20 if has_structure else 0)
        eval_metrics["response_completeness"] = min(completeness, 100)
        
        self.query_evaluations.append(eval_metrics)
        return eval_metrics
    
    def get_evaluation_summary(self) -> dict:
        """Get average evaluation metrics."""
        if not self.query_evaluations:
            return {
                "average_retrieval_quality": 0.0,
                "average_response_relevance": 0.0,
                "average_response_completeness": 0.0,
                "total_queries_evaluated": 0
            }
        
        return {
            "average_retrieval_quality": float(np.mean([e["retrieval_quality"] for e in self.query_evaluations])),
            "average_response_relevance": float(np.mean([e["response_relevance"] for e in self.query_evaluations])),
            "average_response_completeness": float(np.mean([e["response_completeness"] for e in self.query_evaluations])),
            "total_queries_evaluated": len(self.query_evaluations)
        }
