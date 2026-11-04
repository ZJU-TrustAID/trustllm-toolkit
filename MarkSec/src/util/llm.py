# =======================================================
# src/util/llm.py
# Description: Unified Client for LLM APIs (OpenAI, DeepSeek, etc.)
# =======================================================

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Union, List
from openai import OpenAI, APIError
from dotenv import load_dotenv

# Use the centralized logger if available, otherwise default
try:
    from src.util.log import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class LLMConfigurationError(Exception):
    """Exception raised for errors in the LLM configuration."""
    pass

class LLMClient:
    """
    Unified client for interacting with OpenAI-compatible LLM APIs.
    Supports OpenAI (GPT-3.5, GPT-4) and DeepSeek (deepseek-chat, deepseek-v3.2), etc.
    """

    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0.7, 
        system_prompt: str = "You are a helpful assistant.",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: int = 5,
        **kwargs
    ):
        """
        Initialize the LLM Client.

        Args:
            model_name (str): The name of the model (e.g., 'gpt-4', 'deepseek-chat').
            temperature (float): Sampling temperature.
            system_prompt (str): Default system prompt.
            api_key (Optional[str]): API Key. If None, tries to load from environment.
            base_url (Optional[str]): API Base URL. If None, tries to load from environment or uses default.
            max_retries (int): Number of retries for failed requests.
            retry_delay (int): Delay in seconds between retries.
            **kwargs: Additional arguments for the OpenAI client.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        load_dotenv()
        
        # Determine Provider and Configuration
        self.provider = self._infer_provider(model_name, base_url)
        
        # Resolve API Key and Base URL
        self.api_key = api_key or self._resolve_api_key(self.provider)
        self.base_url = base_url or self._resolve_base_url(self.provider)
        
        if not self.api_key:
             logger.warning(f"No API Key found for provider {self.provider}. Requests might fail.")

        logger.info(f"Initializing LLMClient for model: {self.model_name}, provider: {self.provider}, base_url: {self.base_url}")

        # Initialize OpenAI Client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )

    def _infer_provider(self, model_name: str, base_url: str = None) -> str:
        """Infer the provider based on model name or base_url."""
        model_lower = model_name.lower()
        if "deepseek" in model_lower:
            return "deepseek"
        elif "gpt" in model_lower or "o1-" in model_lower:
            return "openai"
        elif base_url and "dashscope" in base_url:
             return "aliyun" # Bailian/Dashscope
        else:
            return "unknown"

    def _resolve_api_key(self, provider: str) -> Optional[str]:
        """Resolve API Key from environment variables based on provider."""
        if provider == "deepseek":
            return os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        elif provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider == "aliyun":
             return os.getenv("DASHSCOPE_API_KEY")
        
        # Fallback check
        return os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

    def _resolve_base_url(self, provider: str) -> Optional[str]:
        """Resolve Base URL from environment variables or defaults."""
        if provider == "deepseek":
            # Check for specific DeepSeek URL env vars first
            url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_API_URL")
            if url: return url
            
            # If DASHSCOPE_API_KEY is present, prefer Aliyun endpoint for DeepSeek (Bailian)
            if os.getenv("DASHSCOPE_API_KEY"):
                return "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return "https://api.deepseek.com"
            
        elif provider == "openai":
            return os.getenv("OPENAI_BASE_URL") # None by default implies official OpenAI
            
        return os.getenv("LLM_BASE_URL") # Generic fallback

    def chat_completion(
        self, 
        messages: list, 
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Execute a chat completion request with retry logic.
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Handle thinking mode for DeepSeek models automatically
        if "deepseek-v3.2" in self.model_name or "deepseek-v3.1" in self.model_name:
             if "extra_body" not in kwargs:
                 kwargs["extra_body"] = {"enable_thinking": True}

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Sending request to {self.model_name} (Attempt {attempt+1})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs
                )
                return response
            except APIError as e:
                error_msg = str(e).lower()
                if "insufficient balance" in error_msg or "insufficient_quota" in error_msg:
                     logger.critical(f"Insufficient Balance/Quota for model {self.model_name}. Stopping retries.")
                     raise e
                
                logger.error(f"LLM API Error (Attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt)) # Exponential backoff
                else:
                    raise e
            except Exception as e:
                logger.error(f"Unexpected LLM Error (Attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise e

    def get_result(self, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple text-to-text generation.
        """
        sys_prompt = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.chat_completion(messages)
        return response.choices[0].message.content

    def get_json_result(self, query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Get structured JSON result.
        Assumes the model supports JSON mode or follows instructions well.
        """
        sys_prompt = system_prompt or self.system_prompt
        # Ensure system prompt encourages JSON
        if "json" not in sys_prompt.lower():
            sys_prompt += " You must respond with valid JSON."
            
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            # Try using response_format={"type": "json_object"} if supported
            response = self.chat_completion(
                messages, 
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.warning(f"JSON mode request failed or parsing failed: {e}. Retrying with manual parsing.")
            # Fallback: simple generation and try to parse
            response = self.chat_completion(messages)
            content = response.choices[0].message.content
            
            # Simple cleanup for markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            return json.loads(content)


# --- Backward Compatibility / Helper Functions ---

def chat(
    message: str,
    history: list = None,
    model: str = "gpt-4",
    system_prompt: str = "You are a research assistant.",
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Standalone chat function for quick usage, compatible with old oracle.py usage.
    """
    if history is None:
        history = [{"role": "system", "content": system_prompt}]
        
    client = LLMClient(model_name=model, temperature=temperature)
    
    # Append new message to history
    messages = history + [{"role": "user", "content": message}]
    
    response = client.chat_completion(messages, **kwargs)
    return response.choices[0].message.content

class DeepSeekAPI:
    """Backward compatibility wrapper for DeepSeekAPI."""
    def __init__(self, model, temperature, system_content):
        self.client = LLMClient(model_name=model, temperature=temperature, system_prompt=system_content)
    
    def get_result(self, query):
        return self.client.get_result(query)

    def get_result_from_deepseek(self, query):
        return self.client.get_result(query)

class OpenAIAPI:
    """Backward compatibility wrapper for OpenAIAPI."""
    def __init__(self, model, temperature, system_content):
        self.client = LLMClient(model_name=model, temperature=temperature, system_prompt=system_content)

    def get_result(self, query):
        return self.client.get_result(query)
    
    def get_result_from_gpt4(self, query):
        # Override model temporary if needed, or just use the configured client
        return self.client.get_result(query)

if __name__ == "__main__":
    # Test
    try:
        client = LLMClient(model_name="deepseek-chat", temperature=0.5)
        print("Testing DeepSeek connection...")
        print(client.get_result("Say hello!"))
    except Exception as e:
        print(f"Test failed: {e}")
