"""
Ollama SDK Helper Functions

This module provides convenient wrapper functions for the Ollama SDK,
including model management, chat, generation, embeddings, and more.
"""

from ollama import (
    Client,
    AsyncClient,
    chat,
    generate,
    embed,
    pull,
    push,
    create,
    delete,
    copy,
    show,
    list as list_models,
    ps,
)
from typing import Dict, List, Any, Optional, Iterator, Union

import asyncio


class OllamaHelper:
    """Main helper class for Ollama operations"""
    
    def __init__(self, host: Optional[str] = None):
        """
        Initialize Ollama helper
        
        Args:
            host: Optional Ollama server host (e.g., 'http://localhost:11434')
        """
        self.client = Client(host=host) if host else Client()
        self.async_client = AsyncClient(host=host) if host else AsyncClient()
    
    # ==================== Model Management ====================
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all installed models
        
        Returns:
            List of model dictionaries with name, size, and details
        """
        try:
            response = list_models()

            return [
                {
                    'name': model.model,
                    'name_short': model.model.replace(":latest", ""),
                    'size': model.size,
                    'size_mb': round(model.size / 1024 / 1024, 2),
                    'modified_at': model.modified_at,
                    'digest': model.digest,
                    'details': {
                        'format': model.details.format if model.details else None,
                        'family': model.details.family if model.details else None,
                        'parameter_size': model.details.parameter_size if model.details else None,
                        'quantization_level': model.details.quantization_level if model.details else None,
                    } if model.details else None
                }
                for model in response.models
            ]
        except Exception as e:
            return []
    
    def get_model_names(self) -> List[str]:
        """
        Get list of installed model names
        
        Returns:
            List of model name strings
        """
        models = self.list_models()
        return [model['name'] for model in models]
    
    def pull_model(self, model_name: str, stream: bool = True) -> Union[Dict, Iterator]:
        """
        Pull/download a model from the Ollama library
        
        Args:
            model_name: Name of the model to pull (e.g., 'gemma3', 'llama3.2')
            stream: Whether to stream progress updates
            
        Returns:
            Progress updates if stream=True, else final response
        """
        return pull(model_name, stream=stream)
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a model from local storage
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Response dictionary
        """
        try:
            response = delete(model_name)
            return {'success': True, 'message': f'Model {model_name} deleted'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a model to a new name
        
        Args:
            source: Source model name
            destination: Destination model name
            
        Returns:
            Response dictionary
        """
        try:
            copy(source, destination)
            return {'success': True, 'message': f'Model copied from {source} to {destination}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_model(self, name: str, from_model: str, system: Optional[str] = None, 
                     modelfile: Optional[str] = None, stream: bool = False) -> Any:
        """
        Create a custom model with specific system prompt or modelfile
        
        Args:
            name: Name for the new model
            from_model: Base model to build from
            system: Optional system prompt
            modelfile: Optional modelfile content
            stream: Whether to stream creation progress
            
        Returns:
            Creation response
        """
        kwargs = {'model': name}
        
        if from_model:
            kwargs['from_'] = from_model
        if system:
            kwargs['system'] = system
        if modelfile:
            kwargs['modelfile'] = modelfile
        kwargs['stream'] = stream
        
        return self.client.create(**kwargs)
    
    def show_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model details dictionary
        """
        try:
            response = show(model_name)
            return {
                'name': model_name,
                'modified_at': response.modified_at,
                'template': response.template,
                'modelfile': response.modelfile,
                'parameters': response.parameters,
                'license': response.license,
                'details': {
                    'format': response.details.format if response.details else None,
                    'family': response.details.family if response.details else None,
                    'families': response.details.families if response.details else None,
                    'parameter_size': response.details.parameter_size if response.details else None,
                    'quantization_level': response.details.quantization_level if response.details else None,
                } if response.details else None,
                'capabilities': response.capabilities,
            }
        except Exception as e:
            return {'error': str(e)}
    
    def list_running_models(self) -> List[Dict[str, Any]]:
        """
        List currently running/loaded models
        
        Returns:
            List of running model details
        """
        try:
            response = ps()
            return [
                {
                    'name': model.model,
                    'size': model.size,
                    'size_mb': round(model.size / 1024 / 1024, 2),
                    'size_vram': model.size_vram,
                    'size_vram_mb': round(model.size_vram / 1024 / 1024, 2),
                    'digest': model.digest,
                    'expires_at': model.expires_at,
                    'context_length': model.context_length,
                }
                for model in response.models
            ]
        except Exception as e:
            return []
    
    # ==================== Chat ====================
    
    def chat(self, model: str, messages: List[Dict[str, Any]], 
             stream: bool = False, **options) -> Any:
        """
        Chat with a model
        
        Args:
            model: Model name
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response
            **options: Additional options (temperature, etc.)
            
        Returns:
            Chat response or stream iterator
        """
        kwargs = {'model': model, 'messages': messages, 'stream': stream}
        if options:
            kwargs['options'] = options
        return chat(**kwargs)
    
    async def async_chat(self, model: str, messages: List[Dict[str, Any]], 
                         stream: bool = False, **options) -> Any:
        """
        Async chat with a model
        
        Args:
            model: Model name
            messages: List of message dictionaries
            stream: Whether to stream the response
            **options: Additional options
            
        Returns:
            Async chat response or stream iterator
        """
        kwargs = {'model': model, 'messages': messages, 'stream': stream}
        if options:
            kwargs['options'] = options
        return await self.async_client.chat(**kwargs)
    
    # ==================== Generate ====================
    
    def generate(self, model: str, prompt: str, stream: bool = False,
                 images: Optional[List] = None, **options) -> Any:
        """
        Generate text from a prompt
        
        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream the response
            images: Optional list of images (for multimodal models)
            **options: Additional options
            
        Returns:
            Generation response or stream iterator
        """
        kwargs = {'model': model, 'prompt': prompt, 'stream': stream}
        if images:
            kwargs['images'] = images
        if options:
            kwargs['options'] = options
        return generate(**kwargs)
    
    async def async_generate(self, model: str, prompt: str, stream: bool = False,
                            images: Optional[List] = None, **options) -> Any:
        """
        Async generate text from a prompt
        
        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream
            images: Optional images
            **options: Additional options
            
        Returns:
            Async generation response or stream iterator
        """
        kwargs = {'model': model, 'prompt': prompt, 'stream': stream}
        if images:
            kwargs['images'] = images
        if options:
            kwargs['options'] = options
        return await self.async_client.generate(**kwargs)
    
    # ==================== Embeddings ====================
    
    def embed(self, model: str, input_text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Generate embeddings for text
        
        Args:
            model: Model name (e.g., 'nomic-embed-text', 'llama3.2')
            input_text: Text string or list of strings
            
        Returns:
            Embeddings response with 'embeddings' key
        """
        return embed(model=model, input=input_text)
    
    async def async_embed(self, model: str, input_text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Async generate embeddings
        
        Args:
            model: Model name
            input_text: Text string or list of strings
            
        Returns:
            Async embeddings response
        """
        return await self.async_client.embed(model=model, input=input_text)
    
    # ==================== Tools / Function Calling ====================
    
    def chat_with_tools(self, model: str, messages: List[Dict[str, Any]], 
                        tools: List, stream: bool = False) -> Any:
        """
        Chat with function calling tools
        
        Args:
            model: Model name
            messages: Chat messages
            tools: List of tool functions or schemas
            stream: Whether to stream
            
        Returns:
            Chat response with potential tool calls
        """
        return chat(model=model, messages=messages, tools=tools, stream=stream)
    
    # ==================== Utility Functions ====================
    
    def is_model_installed(self, model_name: str) -> bool:
        """
        Check if a model is installed
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if installed, False otherwise
        """
        installed_models = self.get_model_names()
        return model_name in installed_models
    
    def get_model_size(self, model_name: str) -> Optional[float]:
        """
        Get model size in MB
        
        Args:
            model_name: Model name
            
        Returns:
            Size in MB or None if not found
        """
        models = self.list_models()
        for model in models:
            if model['name'] == model_name:
                return model['size_mb']
        return None
    
    def ensure_model(self, model_name: str) -> bool:
        """
        Ensure a model is installed, pull if not
        
        Args:
            model_name: Model name
            
        Returns:
            True if model is available, False if pull failed
        """
        if self.is_model_installed(model_name):
            return True
        
        try:
            # Pull the model
            for _ in self.pull_model(model_name, stream=True):
                pass
            return True
        except Exception:
            return False


# Convenience functions for direct use
_helper = OllamaHelper()

def get_installed_models() -> List[str]:
    return _helper.get_model_names()

def get_model_details() -> List[Dict[str, Any]]:
    return _helper.list_models()

def pull_model(model_name: str, stream: bool = True):
    return _helper.pull_model(model_name, stream)

def delete_model(model_name: str) -> Dict[str, Any]:
    return _helper.delete_model(model_name)

def is_model_installed(model_name: str) -> bool:
    return _helper.is_model_installed(model_name)
