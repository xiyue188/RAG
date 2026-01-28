"""
RAG 系统核心模块
"""

from .chunker import chunk_text
from .embedder import Embedder
from .vectordb import VectorDB
from .retriever import Retriever
from .ingestion import DocumentIngestion
from .llm import LLMClient

__all__ = [ 
    "chunk_text",
    "Embedder",
    "VectorDB",
    "Retriever",
    "DocumentIngestion",
    "LLMClient",
]

#定义包的公开 API，使用 from rag import * 时只会导入这些名称，隐藏内部实现细节。

__version__ = "1.0.0"
