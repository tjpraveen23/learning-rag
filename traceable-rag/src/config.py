import os
from pathlib import Path

from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(PROJECT_ROOT / ".env")


# Application
APP_ENV = os.getenv("APP_ENV", "development")


# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


# Paths

EXTRACTED_DATA_PATH = (
    PROJECT_ROOT / os.getenv(
        "EXTRACTED_DATA_PATH",
        "data/extracted",
    )
)
NORMALIZED_DATA_PATH = (
    PROJECT_ROOT / os.getenv(
        "NORMALIZED_DATA_PATH",
        "data/normalized",
    )
)

CHUNKS_DATA_PATH = (
    PROJECT_ROOT / os.getenv(
        "CHUNKS_DATA_PATH",
        "data/chunks",
    )
)


# Document processing
DEFAULT_DOCUMENT_VERSION = os.getenv(
    "DEFAULT_DOCUMENT_VERSION",
    "v1",
)

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# Vector database
CHROMA_PERSIST_DIRECTORY = (
    PROJECT_ROOT / os.getenv(
        "CHROMA_PERSIST_DIRECTORY",
        "data/chroma",
    )
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "traceable_rag",
)

# Retrieval
RETRIEVAL_TOP_K = int(
    os.getenv("RETRIEVAL_TOP_K", "3")
)

RETRIEVAL_MIN_SCORE = float(
    os.getenv("RETRIEVAL_MIN_SCORE", "0.0")
)

# LLM - Groq

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL"
)

GROQ_TEMPERATURE = float(
    os.getenv("GROQ_TEMPERATURE", "0")
)

RETRIEVAL_MAX_DISTANCE = float(
    os.getenv(
        "RETRIEVAL_MAX_DISTANCE",
        "1.55",
    )
)

RETRIEVAL_MIN_CHUNKS = int(
    os.getenv(
        "RETRIEVAL_MIN_CHUNKS",
        "2",
    )
)