# config.py
import os
from typing import Dict, List

class Config:
    """Configuration class for Multi-LOB RAG Framework"""
    
    # Vector Database Configuration
    VECTOR_DB_BASE_PATH = "vector_databases"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # LOB Configuration
    LOB_FOLDERS = {
        "LOB1": "data/lob1_documents",
        "LOB2": "data/lob2_documents", 
        "LOB3": "data/lob3_documents"
    }
    
    # Active LOB for queries (can be changed via config)
    ACTIVE_LOB = "LOB1"
    
    # Supported file types
    SUPPORTED_FILE_TYPES = ['.pdf', '.txt', '.xlsx', '.xls', '.docx']
    
    # LLM Configuration
    LLM_TYPE = "openai"  # Options: 'inhouse' or 'openai'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Only needed for OpenAI validation
    
    @classmethod
    def get_vector_db_name(cls, lob_name: str) -> str:
        """Generate vector database name for a LOB"""
        return f"{lob_name}_VectorDB"
    
    @classmethod
    def get_vector_db_path(cls, lob_name: str) -> str:
        """Generate vector database path for a LOB"""
        return os.path.join(cls.VECTOR_DB_BASE_PATH, cls.get_vector_db_name(lob_name))
