# OpenAILLM.py
import logging
from typing import Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAILLM:
    """OpenAI LLM implementation with sendOpenAI method for validation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def sendOpenAI(self, inputPrompt: str) -> str:
        """
        Send prompt to OpenAI GPT-4o-mini model and return result
        
        Args:
            inputPrompt: The input prompt string
            
        Returns:
            str: The model's response
        """
        try:
            logger.info("Processing prompt with OpenAI GPT-4o-mini...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": inputPrompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            logger.info("OpenAI processing completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in sendOpenAI: {str(e)}")
            return f"Error processing OpenAI request: {str(e)}"
