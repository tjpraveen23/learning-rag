# document_processor.py
import os
import logging
from typing import List, Dict
from pathlib import Path

from langchain.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredExcelLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing for multiple file types"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.supported_loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.docx': Docx2txtLoader
        }
    
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Load all supported documents from a folder"""
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.warning(f"Folder {folder_path} does not exist")
            return documents
        
        for file_path in folder.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_loaders:
                try:
                    loader_class = self.supported_loaders[file_path.suffix.lower()]
                    loader = loader_class(str(file_path))
                    file_documents = loader.load()
                    
                    # Add metadata
                    for doc in file_documents:
                        doc.metadata.update({
                            'source_file': str(file_path),
                            'file_type': file_path.suffix,
                            'folder': folder_path
                        })
                    
                    documents.extend(file_documents)
                    logger.info(f"Loaded {len(file_documents)} documents from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return []
