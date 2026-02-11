"""
数据模型定义（Pydantic v2）
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# ==================== 请求模型 ====================

class QueryRequest(BaseModel):
    """对话查询请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    enable_multi_query: bool = Field(True, description="是否启用多查询扩展")
    enable_rerank: bool = Field(False, description="是否启用重排序")
    enable_hybrid: bool = Field(True, description="是否启用混合检索")
    enable_citation: bool = Field(True, description="是否启用引用追踪")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "什么是宠物政策？",
                "enable_multi_query": True,
                "enable_rerank": False,
                "enable_hybrid": True,
                "enable_citation": True
            }]
        }
    }

# ==================== 响应模型 ====================

class CitationInfo(BaseModel):
    """引用信息"""
    doc_id: str = Field(..., description="文档ID")
    file: str = Field(..., description="文件名")
    category: str = Field(..., description="分类")
    content: str = Field(..., description="引用内容")
    score: float = Field(..., description="相似度分数")

class QueryResponse(BaseModel):
    """对话查询响应"""
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="原始问题")
    resolved_question: Optional[str] = Field(None, description="消解后的问题")
    answer: str = Field(..., description="回答内容")
    citations: List[CitationInfo] = Field(default_factory=list, description="引用列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    total: int = Field(..., description="总数")

# ==================== 错误响应模型 ====================

class ErrorDetail(BaseModel):
    """错误详情"""
    message: str = Field(..., description="错误消息")
    type: str = Field(..., description="错误类型")
    detail: Optional[str] = Field(None, description="详细信息")

class ErrorResponse(BaseModel):
    """统一错误响应"""
    error: ErrorDetail = Field(..., description="错误详情")
    status_code: int = Field(..., description="HTTP状态码")
    path: Optional[str] = Field(None, description="请求路径")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="时间戳")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "RateLimitError",
                    "detail": "Too many requests. Please try again later."
                },
                "status_code": 429,
                "path": "/api/v1/chat/message",
                "timestamp": "2026-02-11T12:00:00Z"
            }
        }
    }
