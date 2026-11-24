# ==========================================
# ADVANCED RAG ENGINEERING LAB
# Project Configuration
# ==========================================

# Import os to work with environment variables
import os

# Load API keys and settings from the .env file
from dotenv import load_dotenv

# Read the .env file
load_dotenv()


# ==========================================
# AI MODEL SETTINGS
# ==========================================

# Select which AI provider we want to use
# Options: "gemini" or "groq"

LLM_PROVIDER = "gemini"


# Read Gemini API key from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Read Groq API key from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================
# EMBEDDING SETTINGS
# ==========================================

# We will initially use a local Hugging Face model
# This model converts text into numerical vectors

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# VECTOR DATABASE SETTINGS
# ==========================================

# ChromaDB will store our document embeddings

CHROMA_DB_PATH = "./embeddings/chroma_db"


# ==========================================
# DOCUMENT SETTINGS
# ==========================================

# Folder containing our sample documents

DOCUMENTS_PATH = "./documents"


# ==========================================
# PROJECT SETTINGS
# ==========================================

PROJECT_NAME = "Advanced RAG Engineering Lab"

print("Project configuration loaded successfully!")