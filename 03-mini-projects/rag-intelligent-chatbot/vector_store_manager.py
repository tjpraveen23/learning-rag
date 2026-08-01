import os
import logging
from typing import List, Optional, Dict
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector stores for multiple LOBs"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_stores: Dict[str, Chroma] = {}
    
    def create_or_load_vector_store(self, lob_name: str, documents: List[Document] = None) -> Optional[Chroma]:
        """Create or load vector store for a LOB"""
        vector_db_path = Config.get_vector_db_path(lob_name)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(vector_db_path, exist_ok=True)
            
            if documents and len(documents) > 0:
                # Create new vector store with documents
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=vector_db_path
                )
                vector_store.persist()
                logger.info(f"Created vector store for {lob_name} with {len(documents)} documents")
            else:
                # Load existing vector store
                vector_store = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store for {lob_name}")
            
            self.vector_stores[lob_name] = vector_store
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating/loading vector store for {lob_name}: {str(e)}")
            return None
    
    def get_vector_store(self, lob_name: str) -> Optional[Chroma]:
        """Get vector store for a LOB"""
        if lob_name in self.vector_stores:
            return self.vector_stores[lob_name]
        
        # Try to load existing vector store
        return self.create_or_load_vector_store(lob_name)
    
    def search_similar_documents(self, lob_name: str, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in LOB vector store"""
        vector_store = self.get_vector_store(lob_name)
        
        if not vector_store:
            logger.warning(f"No vector store found for {lob_name}")
            return []
        
        try:
            results = vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} similar documents for {lob_name}")
            return results
        except Exception as e:
            logger.error(f"Error searching documents for {lob_name}: {str(e)}")
            return []
