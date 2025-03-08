import os
import time
import hashlib
import logging
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
    
    # Available models with their specific configurations
    AVAILABLE_MODELS = {
        "claude-3-sonnet": {
            "model": "anthropic/claude-3-7-sonnet-latest",
            "provider": "anthropic",
            "supports_temperature": True,
            "token_param": "max_tokens"
        },
        "gpt-4": {
            "model": "openai/gpt-4o",
            "provider": "openai",
            "supports_temperature": True,
            "token_param": "max_tokens"
        },
        "deepseek-v3": {
            "model": "deepinfra/deepseek-ai/DeepSeek-V3",
            "provider": "deepseek",
            "supports_temperature": True,
            "token_param": "max_tokens"
        },
        "deepseek-r1": {
            "model": "deepinfra/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "provider": "deepseek",
            "supports_temperature": True,
            "token_param": "max_tokens"
        },
        "o3-mini": {
            "model": "openai/o3-mini",
            "provider": "openai",
            "supports_temperature": False,
            "token_param": "max_completion_tokens"
        }
    }
    
    def __init__(self, api_key: str = None, default_model: str = None):
        """
        Initializes the Requesty API Client
        
        Parameters:
        api_key (str, optional): Requesty API key (defaults to REQUESTY_API_KEY env var)
        default_model (str, optional): Default model (defaults to DEFAULT_MODEL env var or o3-mini)
        """
        # Initialize response cache
        self._response_cache = {}
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv('REQUESTY_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or in the .env file as REQUESTY_API_KEY")
            
        # Configure OpenAI with Requesty base URL
        openai.api_key = self.api_key
        openai.api_base = "https://router.requesty.ai/v1"
        
        # Use provided model or get from environment, with fallback to o3-mini
        self.default_model = default_model or os.getenv('DEFAULT_MODEL', 'o3-mini')
        
        if self.default_model not in self.AVAILABLE_MODELS:
            logging.warning(f"Invalid model: {self.default_model}. Falling back to o3-mini")
            self.default_model = 'o3-mini'
        
        # Log the selected model
        logging.info(f"Using model: {self.default_model} ({self.AVAILABLE_MODELS[self.default_model]['model']})")
    
    def generate_response(self, prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000, use_cache: bool = True) -> str:
        """
        Generates a response with the selected model
        
        Parameters:
        prompt (str): Prompt for the model
        model (str, optional): Model to use (overrides default_model)
        temperature (float): Creativity of the response (0.0-1.0) - not used for all models
        max_tokens (int): Maximum number of tokens in the response
        use_cache (bool): Whether to use cached responses
        
        Returns:
        str: Generated response
        """
        model_name = model if model in self.AVAILABLE_MODELS else self.default_model
        model_info = self.AVAILABLE_MODELS[model_name]
        
        # Create cache key using a hash of prompt and parameters
        if use_cache:
            cache_key = hashlib.md5(f"{prompt[:100]}_{model_name}_{temperature}_{max_tokens}".encode()).hexdigest()
            
            # Check if response is cached
            if cache_key in self._response_cache:
                logging.info(f"Using cached response for query")
                return self._response_cache[cache_key]
        
        try:
            # Log request info
            logging.info(f"Making API request to {model_info['model']}")
            start_time = time.time()
            
            # Prepare parameters based on model requirements
            params = {
                "model": model_info["model"],
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "provider": model_info["provider"],
                "request_timeout": 180  # 3-minute timeout
            }
            
            # Add temperature if supported by the model
            if model_info["supports_temperature"]:
                params["temperature"] = temperature
                
            # Add token limit with the correct parameter name for this model
            token_param = model_info["token_param"]
            params[token_param] = max_tokens
            
            # Create the completion using the OpenAI client
            response = openai.ChatCompletion.create(**params)
            
            # Extract the response from the API response
            response_text = response.choices[0].message.content
            
            # Log timing information
            elapsed_time = time.time() - start_time
            token_count = len(response_text.split())
            tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
            logging.info(f"API request completed in {elapsed_time:.2f} seconds ({tokens_per_sec:.2f} tokens/sec)")
            
            # Cache the response if caching is enabled
            if use_cache:
                self._response_cache[cache_key] = response_text
                
            return response_text
            
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
