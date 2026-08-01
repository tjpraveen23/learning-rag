# llm_factory.py
import logging
from typing import Optional, Union
from InhouseLLM import InhouseLLM
from OpenAILLM import OpenAILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory class to create and manage different LLM implementations"""
    
    @staticmethod
    def create_llm(llm_type: str, **kwargs) -> Union[InhouseLLM, OpenAILLM]:
        """
        Create LLM instance based on type
        
        Args:
            llm_type: Type of LLM ('inhouse' or 'openai')
            **kwargs: Additional arguments for LLM initialization
            
        Returns:
            LLM instance
        """
        if llm_type.lower() == 'inhouse':
            return InhouseLLM()
        elif llm_type.lower() == 'openai':
            api_key = kwargs.get('api_key')
            return OpenAILLM(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

class LLMClient:
    """Unified client for different LLM implementations"""
    
    def __init__(self, llm_type: str, **kwargs):
        """
        Initialize LLM client
        
        Args:
            llm_type: Type of LLM ('inhouse' or 'openai')
            **kwargs: Additional arguments for LLM initialization
        """
        self.llm_type = llm_type
        self.llm = LLMFactory.create_llm(llm_type, **kwargs)
        logger.info(f"LLM Client initialized with {llm_type} model")
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """
        Generate response using the configured LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response or None if error
        """
        try:
            if self.llm_type.lower() == 'inhouse':
                return self.llm.sendTachyon(prompt)
            elif self.llm_type.lower() == 'openai':
                return self.llm.sendOpenAI(prompt)
            else:
                logger.error(f"Unknown LLM type: {self.llm_type}")
                return None
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None
