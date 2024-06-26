import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
import httpx
import structlog
from langchain_anthropic import ChatAnthropic


logger = structlog.get_logger(__name__)

@lru_cache(maxsize=1)
def get_anthropic_llm(temperature: float = 0, max_tokens: int = 2000):
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        max_tokens_to_sample=max_tokens,
        temperature=temperature,
    )
    return model
