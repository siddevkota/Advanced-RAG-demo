"""Configuration for Streamlit frontend."""
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# UI Configuration
PAGE_TITLE = "Advanced RAG Chatbot"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Theme
THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f0f2f6",
    "textColor": "#262730"
}
