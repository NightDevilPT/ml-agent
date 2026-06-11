"""
Multi-Provider LLM Integration Utility
======================================

Connects to either Docker-based LLM containers or Google Gemini models.

Usage:
    from utils.llm import get_llm
    
    # For Docker LLM
    llm = get_llm(provider="docker", context=4)
    
    # For Google Gemini
    llm = get_llm(provider="gemini", context=4)
    
    response = llm.invoke("Write a TypeScript function")
"""

import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
load_dotenv()

# Docker LLM Configuration
DOCKER_LLM_BASE_URL = os.getenv("DOCKER_LLM_BASE_URL", "http://localhost:12434/v1")
DOCKER_LLM_MODEL = os.getenv("DOCKER_LLM_MODEL", "ai/qwen2.5")
DOCKER_LLM_API_KEY = os.getenv("DOCKER_LLM_API_KEY", "not-needed")

# Google Gemini Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY"))

# Default provider
api_key_set = bool(GEMINI_API_KEY)
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


def get_llm(
    provider: Literal["docker", "gemini"] = "gemini",
    context: int = 4,
    temperature: float = 0.1
):
    """
    Create and return LangChain-compatible LLM client.
    Supports both Docker-based local LLMs and Google Gemini.

    Parameters
    ----------
    provider : str, optional
        LLM provider to use: "docker" or "gemini".
        If None, uses LLM_PROVIDER environment variable or defaults to "docker".
    context : int
        Context window size in multiples of 1024 tokens.
        4 = 4096, 8 = 8192, 16 = 16384. Default 4.
    temperature : float
        Response randomness. Lower = more deterministic. Default 0.1.

    Returns
    -------
    ChatOpenAI or ChatGoogleGenerativeAI
        LangChain-compatible LLM client.

    Raises
    ------
    ValueError
        If invalid provider specified or if GEMINI_API_KEY is missing for Gemini.
        If provider is unrecognized.

    Examples
    --------
    >>> # Docker LLM
    >>> llm = get_llm(provider="docker", context=8)
    
    >>> # Google Gemini
    >>> llm = get_llm(provider="gemini", context=8)
    
    >>> # Using default provider from environment
    >>> llm = get_llm(context=4)
    
    >>> response = llm.invoke("Write a Jest test for this function...")
    """
    # Determine provider
    if provider is None:
        provider = DEFAULT_PROVIDER
    
    provider = provider.lower().strip()
    
    if provider == "docker":
        return ChatOpenAI(
            model=DOCKER_LLM_MODEL,
            base_url=DOCKER_LLM_BASE_URL,
            api_key=DOCKER_LLM_API_KEY,
            temperature=temperature,
            max_tokens=context * 1024,
        )
    
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it to your Google AI Studio API key."
            )
        
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
            max_output_tokens=context * 1024,
        )
    
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Supported providers: 'docker', 'gemini'"
        )


def list_available_providers() -> dict:
    """
    Check which LLM providers are available based on configuration.
    
    Returns
    -------
    dict
        Dictionary with provider names as keys and availability status as values.
    
    Examples
    --------
    >>> list_available_providers()
    {'docker': True, 'gemini': False}
    """
    return {
        "docker": True,  # Always available (assumes Docker is running)
        "gemini": bool(GEMINI_API_KEY)
    }



def extract_token_usage(response: AIMessage) -> int:
    """
    Extract total token usage safely from raw AIMessage metadata.
    Shared utility used across all LangGraph nodes.
    """
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            return response.usage_metadata.get("total_tokens", 0)
        if hasattr(response, "response_metadata") and response.response_metadata:
            return response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    except Exception:
        pass
    return 0
