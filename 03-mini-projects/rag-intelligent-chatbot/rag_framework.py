# rag_framework.py
import logging
from typing import List, Optional, Dict
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from llm_factory import LLMClient
from system_prompt import SystemPrompt
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLOBRAGFramework:
    """Main RAG Framework supporting multiple LOBs"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_client = LLMClient(
            Config.LLM_TYPE,
            api_key=Config.OPENAI_API_KEY if Config.LLM_TYPE == 'openai' else None
        )
        self.initialized_lobs = set()
    
    def initialize_lob(self, lob_name: str, force_rebuild: bool = False) -> bool:
        """Initialize a LOB by processing documents and creating vector store"""
        if lob_name not in Config.LOB_FOLDERS:
            logger.error(f"LOB {lob_name} not configured")
            return False
        
        if lob_name in self.initialized_lobs and not force_rebuild:
            logger.info(f"LOB {lob_name} already initialized")
            return True
        
        try:
            folder_path = Config.LOB_FOLDERS[lob_name]
            
            # Load and process documents
            logger.info(f"Processing documents for {lob_name} from {folder_path}")
            documents = self.document_processor.load_documents_from_folder(folder_path)
            
            if not documents:
                logger.warning(f"No documents found for {lob_name}")
                return False
            
            # Split documents into chunks
            chunks = self.document_processor.split_documents(documents)
            
            # Create vector store
            vector_store = self.vector_store_manager.create_or_load_vector_store(
                lob_name, chunks
            )
            
            if vector_store:
                self.initialized_lobs.add(lob_name)
                logger.info(f"Successfully initialized {lob_name}")
                return True
            else:
                logger.error(f"Failed to create vector store for {lob_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing {lob_name}: {str(e)}")
            return False
    
    def initialize_all_lobs(self, force_rebuild: bool = False) -> Dict[str, bool]:
        """Initialize all configured LOBs"""
        results = {}
        for lob_name in Config.LOB_FOLDERS.keys():
            results[lob_name] = self.initialize_lob(lob_name, force_rebuild)
        return results
    
    def query(self, question: str, lob_name: str = None, max_context_docs: int = 5) -> Dict:
        """Process a query for specified LOB"""
        if not lob_name:
            lob_name = Config.ACTIVE_LOB
        
        if lob_name not in self.initialized_lobs:
            logger.warning(f"LOB {lob_name} not initialized. Attempting to initialize...")
            if not self.initialize_lob(lob_name):
                return {
                    'success': False,
                    'error': f'Failed to initialize LOB {lob_name}',
                    'lob': lob_name,
                    'question': question
                }
        
        try:
            # Retrieve relevant documents from vector store
            similar_docs = self.vector_store_manager.search_similar_documents(
                lob_name, question, max_context_docs
            )
            
            if not similar_docs:
                return {
                    'success': False,
                    'error': 'No relevant documents found',
                    'lob': lob_name,
                    'question': question
                }
            
            # Build context from retrieved documents
            context = self._build_context(similar_docs)
            
            # Build complete prompt
            prompt = SystemPrompt.build_prompt(context, question, lob_name)
            
            # Get response from in-house LLM
            response = self.llm_client.generate_response(prompt)
            
            if response:
                return {
                    'success': True,
                    'answer': response,
                    'lob': lob_name,
                    'question': question,
                    'context_sources': [doc.metadata.get('source_file', 'Unknown') for doc in similar_docs],
                    'num_context_docs': len(similar_docs)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get response from LLM',
                    'lob': lob_name,
                    'question': question
                }
                
        except Exception as e:
            logger.error(f"Error processing query for {lob_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'lob': lob_name,
                'question': question
            }
    
    def _build_context(self, documents: List) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source_file', 'Unknown source')
            content = doc.page_content.strip()
            context_parts.append(f"Document {i} (Source: {source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def get_available_lobs(self) -> List[str]:
        """Get list of available LOBs"""
        return list(Config.LOB_FOLDERS.keys())
    
    def get_initialized_lobs(self) -> List[str]:
        """Get list of initialized LOBs"""
        return list(self.initialized_lobs)
