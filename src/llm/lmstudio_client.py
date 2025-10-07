from langchain_openai import ChatOpenAI
import os
from typing import List, Dict

def get_llm() -> ChatOpenAI:
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    model = os.getenv('LMSTUDIO_MODEL', "qwen/qwen3-4b-2507")
    temperature = 0.2
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        model=model
    )
    
    return llm

