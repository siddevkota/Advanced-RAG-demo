"""Clean RAG Chatbot Interface"""
import streamlit as st
import requests
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import app_config
import api_client

API_URL = app_config.API_BASE_URL
client = api_client.APIClient(API_URL)

st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fixed input at bottom
st.markdown("""
<style>
    .main {padding-bottom: 100px;}
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
    .chat-container {max-width: 800px; margin: 0 auto;}
    .query-info {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        border-left: 3px solid #007bff;
    }
    .eval-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background: #e9ecef;
        border-radius: 0.3rem;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()
if "config" not in st.session_state:
    st.session_state.config = {
        "chunk_size": 400,
        "chunk_overlap": 80,
        "stage1_k": 30,
        "top_k": 5
    }

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Health check
    try:
        health = client.health_check()
        if health.get("status") == "healthy":
            st.success("Backend Connected")
            st.caption(f"Provider: {health.get('provider', 'N/A')}")
            st.caption(f"Documents: {health.get('total_documents', 0)}")
            st.caption(f"Chunks: {health.get('total_chunks', 0)}")
        else:
            st.error("Backend Offline")
            st.stop()
    except:
        st.error("Cannot connect to backend")
        st.stop()
    
    st.divider()
    
    # PDF Upload
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Add documents to knowledge base", 
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process PDFs"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Prepare files for upload
                files = [
                    ("files", (file.name, file.getvalue(), "application/pdf"))
                    for file in uploaded_files
                ]
                
                status_text.text(f"Uploading {len(uploaded_files)} file(s)...")
                progress_bar.progress(0.3)
                
                # Upload PDFs
                response = requests.post(f"{API_URL}/upload_pdf", files=files)
                
                if response.ok:
                    result = response.json()
                    progress_bar.progress(1.0)
                    
                    st.success(f"✓ Processed {len(result['files'])} file(s)")
                    st.info(f"Total: {result['total_chunks']} chunks from {result['total_documents']} documents")
                    
                    # Show individual file results
                    with st.expander("File Details"):
                        for file_result in result['files']:
                            if file_result['success']:
                                st.write(f"✓ {file_result['filename']}: {file_result['pages']} pages")
                            else:
                                st.write(f"✗ {file_result['filename']}: {file_result.get('error', 'Failed')}")
                    
                    st.rerun()
                else:
                    st.error("Upload failed")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    st.divider()
    
    # RAG Parameters
    st.subheader("RAG Parameters")
    chunk_size = st.slider("Chunk Size", 100, 1000, st.session_state.config["chunk_size"], 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, st.session_state.config["chunk_overlap"], 10)
    stage1_k = st.slider("Initial Retrieval", 5, 100, st.session_state.config["stage1_k"], 5)
    top_k = st.slider("Final Chunks", 1, 20, st.session_state.config["top_k"], 1)
    
    if st.button("Apply Config"):
        new_config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "stage1_k": stage1_k,
            "top_k": top_k
        }
        
        with st.spinner("Updating configuration..."):
            try:
                result = client.update_config(new_config)
                st.session_state.config = new_config
                st.success(result.get("message", "Config updated"))
            except Exception as e:
                st.error(f"Failed to update config: {e}")
    
    st.divider()
    
    # LLM Settings
    st.subheader("LLM Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    st.divider()
    
    # Statistics
    st.subheader("Statistics")
    try:
        stats = client.get_statistics()
        if stats.get("total_feedback", 0) > 0:
            st.metric("Feedback", stats["total_feedback"])
            st.metric("Satisfaction", f"{stats['satisfaction_rate']:.0f}%")
        else:
            st.info("No feedback yet")
    except:
        pass
    
    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.feedback_given = set()
        st.rerun()

# Main Chat Area
st.title("Advanced RAG Chatbot")
    
    # Display chat messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show details for assistant messages
        if msg["role"] == "assistant" and "details" in msg:
            details = msg["details"]
            
            # Query Rewriting
            if details.get("rewritten_queries"):
                with st.expander("Query Variations", expanded=False):
                    for i, q in enumerate(details["rewritten_queries"], 1):
                        st.caption(f"{i}. {q}")
            
            # Retrieval Info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Stage 1: {details.get('stage1_candidates', 0)} docs")
            with col2:
                st.caption(f"Stage 2: {details.get('stage2_reranked', 0)} docs")
            with col3:
                if details.get('retrieval_scores'):
                    avg_score = sum(details['retrieval_scores']) / len(details['retrieval_scores'])
                    st.caption(f"Avg Score: {avg_score:.2f}")
            
            # Sources/References
            if details.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for source in details["sources"]:
                        chunks_text = f" ({source.get('chunk_count', 1)} chunks)" if source.get('chunk_count', 1) > 1 else ""
                        st.markdown(f"**[Source {source['id']}]** {source['source']} (Page {source['page']}){chunks_text}")
                        st.caption(source['content'])
                        st.divider()
            
            # Reranking Scores
            if details.get("retrieval_scores"):
                with st.expander("Reranking Details"):
                    for i, score in enumerate(details["retrieval_scores"], 1):
                        st.progress(min(abs(score) / 5, 1.0), text=f"Chunk {i}: {score:.3f}")
            
            # Evaluation
            if details.get("evaluation"):
                eval_data = details["evaluation"]
                with st.expander("Quality Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retrieval", f"{eval_data.get('retrieval_quality', 0):.0f}%")
                    with col2:
                        st.metric("Relevance", f"{eval_data.get('response_relevance', 0):.0f}%")
                    with col3:
                        st.metric("Complete", f"{eval_data.get('response_completeness', 0):.0f}%")
            
            # Memory Context
            if details.get("memory_context"):
                with st.expander("Conversation Memory"):
                    st.caption(details["memory_context"])
            
            # Feedback buttons
            if idx not in st.session_state.feedback_given:
                col1, col2, col3 = st.columns([1, 1, 8])
                
                with col1:
                    if st.button("👍", key=f"up_{idx}", use_container_width=True):
                        try:
                            client.submit_feedback(
                                query=st.session_state.messages[idx-1]["content"],
                                response=msg["content"],
                                rating="👍",
                                comment=None
                            )
                            st.session_state.feedback_given.add(idx)
                            st.success("Thanks!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col2:
                    if st.button("👎", key=f"down_{idx}", use_container_width=True):
                        st.session_state[f"show_comment_{idx}"] = True
                
                # Show comment form if thumbs down clicked
                if st.session_state.get(f"show_comment_{idx}", False):
                    comment = st.text_input(
                        "Optional: What could be better?",
                        key=f"comment_{idx}",
                        placeholder="Leave blank to skip..."
                    )
                    
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        if st.button("Submit", key=f"submit_{idx}", use_container_width=True):
                            try:
                                client.submit_feedback(
                                    query=st.session_state.messages[idx-1]["content"],
                                    response=msg["content"],
                                    rating="👎",
                                    comment=comment if comment else None
                                )
                                st.session_state.feedback_given.add(idx)
                                st.session_state[f"show_comment_{idx}"] = False
                                st.success("Feedback recorded")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    with col_b:
                        if st.button("Cancel", key=f"cancel_{idx}"):
                            st.session_state[f"show_comment_{idx}"] = False
                            st.rerun()

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = client.query(prompt, temperature)
                
                # Display response
                st.markdown(result["response"])
                
                # Store message with details
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "details": result
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
