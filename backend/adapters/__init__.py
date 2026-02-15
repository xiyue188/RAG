"""
适配器层
将核心 RAG 模块适配到 Web API
"""

from .streaming_ingestion import StreamingIngestionAdapter

__all__ = ["StreamingIngestionAdapter"]
