import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
import httpx
import structlog
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq


logger = structlog.get_logger(__name__)

@lru_cache(maxsize=4)
def get_anthropic_llm(model: str):
    llm = ChatAnthropic(
        model_name=model,
        max_tokens_to_sample=2000,
        temperature=0,
    )
    return llm

@lru_cache(maxsize=4)
def get_groq_llm(model: str):
    llm = ChatGroq(
        model_name=model,
        max_tokens=2000,
        temperature=0,
    )
    return llm