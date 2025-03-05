import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import openai
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RequestyLLMClient:
    """
    Client for the Requesty API with support for various models
    """
    
    # Available models
    AVAILABLE_MODELS = {
        "claude-3-sonnet": {
            "model": "anthropic/claude-3-7-sonnet-latest",
            "provider": "anthropic"
        },
        "gpt-4": {
            "model": "openai/gpt-4o",
            "provider": "openai"
        },
        "deepseek-v3": {
            "model": "deepinfra/deepseek-ai/DeepSeek-V3",
            "provider": "deepseek"
        }
    }
    
    def __init__(self, api_key: str = None, default_model: str = None):
        """
        Initializes the Requesty API Client
        
        Parameters:
        api_key (str, optional): Requesty API key (defaults to REQUESTY_API_KEY env var)
        default_model (str, optional): Default model (defaults to DEFAULT_MODEL env var or deepseek-v3)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv('REQUESTY_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or in the .env file as REQUESTY_API_KEY")
            
        # Configure OpenAI with Requesty base URL
        openai.api_key = self.api_key
        openai.api_base = "https://router.requesty.ai/v1"
        
        # Use provided model or get from environment, with fallback to deepseek-v3
        self.default_model = default_model or os.getenv('DEFAULT_MODEL', 'deepseek-v3')
        
        if self.default_model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {self.default_model}. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}")
    
    def generate_response(self, prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generates a response with the selected model
        
        Parameters:
        prompt (str): Prompt for the model
        model (str, optional): Model to use (overrides default_model)
        temperature (float): Creativity of the response (0.0-1.0)
        max_tokens (int): Maximum number of tokens in the response
        
        Returns:
        str: Generated response
        """
        model_name = model if model in self.AVAILABLE_MODELS else self.default_model
        model_info = self.AVAILABLE_MODELS[model_name]
        
        try:
            # Create the completion using the OpenAI client
            response = openai.ChatCompletion.create(
                model=model_info["model"],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                provider=model_info["provider"]  # Add provider as a parameter
            )
            
            # Extract the response from the API response
            return response.choices[0].message.content
            
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {str(e)}")
            return f"OpenAI API error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return f"Unexpected error: {str(e)}"


def example_api_call(prompt: str) -> str:
    """
    Example function for API call
    In practice, you would implement your specific API logic here
    
    Parameters:
    prompt (str): Prepared prompt
    
    Returns:
    str: API response
    """
    try:
        # Placeholder for API call
        print("API call with prompt:")
        print(prompt)
        return "This would be a generated response from the LLM."
    except Exception as e:
        logging.error(f"Error in example API call: {str(e)}")
        return f"Error in example API call: {str(e)}"
