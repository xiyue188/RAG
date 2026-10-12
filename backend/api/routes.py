"""REST API 路由（非流式）

**这一层只做协议的事**：解析请求参数、调用 service、组装响应。
不碰 ChromaDB、不做数据变换、不 import `rag` —— 文档的列举/分组/删除
都在 `services/document_service.py`。

判据是"换成 CLI 还需不需要这段代码"：按文件分组 chunk 不需要 HTTP，
所以它不属于这里。`tests/test_architecture.py` 会检查
`backend/api/` 不得 import `rag`。
"""

import logging

from fastapi import APIRouter, HTTPException
from backend.schemas import (
    QueryRequest, QueryResponse, DocumentListResponse, CitationInfo,
    DocumentInfo, ChunkInfo, ThresholdConfig,
)
from backend.services.chat_service import get_chat_service
from backend.services.document_service import get_document_service

# 用标准库 logging 而非 rag.logger：协议层不 import rag。
# 日志格式由 rag.logger 在应用启动时统一配置到 root logger，
# 这里拿到的 logger 会继承那份配置。
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/config/thresholds", response_model=ThresholdConfig,
            summary="获取相关性阈值")
async def get_thresholds():
    """相关性阈值，供前端配色与分档使用。

    **为什么要有这个端点**：阈值是领域规则的参数，只能有一份。此前前端
    两个组件各自硬编码了一份且值不同（`50` 与 `0.75`），而后端实际用的是
    第三个值 —— 同一个 relevance 在仪表盘显示"足以支撑回答"、在引用卡片
    显示警告色，后端却判定不可答。跨进程的常量副本没有编译器会管，
    只能运行时下发。

    不做鉴权也不缓存：这两个数不是秘密（README 里就写着），
    而且前端每次启动只拉一次。
    """
    from config import ANSWERABLE_MIN_RELEVANCE, RETRIEVAL_MIN_RELEVANCE

    return ThresholdConfig(
        retrieval_min=RETRIEVAL_MIN_RELEVANCE,
        answerable_min=ANSWERABLE_MIN_RELEVANCE,
    )


@router.post("/chat/message", response_model=QueryResponse, summary="发送对话消息（非流式）")
async def send_message(request: QueryRequest):
    """
    非流式对话接口，收集完整结果后返回。
    """
    try:
        chat_service = get_chat_service()
        session_id = chat_service.ensure_session_id(request.session_id)

        resolved_question = None
        answer_chunks = []
        citations = []

        async for chunk in chat_service.answer_stream(
            session_id=session_id,
            question=request.question,
            use_retrieval=request.use_retrieval,
            enable_multi_query=request.enable_multi_query,
            enable_rerank=request.enable_rerank,
            enable_hybrid=request.enable_hybrid,
            enable_citation=request.enable_citation
        ):
            if chunk["type"] == "resolved":
                resolved_question = chunk["data"]["resolved"]
            elif chunk["type"] == "answer_chunk":
                answer_chunks.append(chunk["data"]["content"])
            elif chunk["type"] == "citations":
                citations = [CitationInfo(**c) for c in chunk["data"]["citations"]]

        return QueryResponse(
            session_id=session_id,
            question=request.question,
            resolved_question=resolved_question,
            answer="".join(answer_chunks),
            citations=citations,
        )

    except Exception as e:
        logger.error(f"消息处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history/{session_id}", summary="获取会话历史")
async def get_history(session_id: str):
    try:
        chat_service = get_chat_service()
        history = await chat_service.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"获取历史错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/session/{session_id}", summary="清空会话")
async def clear_session(session_id: str):
    try:
        chat_service = get_chat_service()
        success = await chat_service.clear_session(session_id)
        if success:
            return {"message": "会话已清空", "session_id": session_id}
        raise HTTPException(status_code=404, detail="会话不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空会话错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文档 ====================
#
# 这四个端点原先直接持有 VectorDB、调 ChromaDB、并内联了两段
# "按文件分组 chunk" 的逻辑（共约 60 行）。问题有三：
#   1. 分组逻辑换成 CLI 一字不变，是应用逻辑不是 HTTP 逻辑
#   2. 每个请求新建一次 VectorDB()，重复初始化客户端
#   3. ChromaDB 的 Python 客户端是同步的，直接在 async def 里调
#      会占住事件循环，其中列举文档那条还会拉全库正文
# 现在全部收敛到 services/document_service.py，并 offload 到线程池。


@router.get("/documents", response_model=DocumentListResponse, summary="获取文档列表")
async def list_documents(include_chunks: bool = False):
    """获取文档列表。

    `include_chunks=True` 会拉全库正文，代价随语料线性增长，
    默认关闭 —— 前端按需再调 `/documents/{file}/chunks`。
    """
    try:
        documents = await get_document_service().list_documents(include_chunks)
        return DocumentListResponse(
            documents=[DocumentInfo(**d) for d in documents],
            total=len(documents),
        )
    except Exception as e:
        logger.error(f"获取文档列表错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{file_name:path}/chunks", response_model=list[ChunkInfo],
            summary="获取文档切片")
async def get_document_chunks(file_name: str):
    """按需获取指定文档的切片内容。"""
    try:
        chunks = await get_document_service().get_chunks(file_name)
        return [ChunkInfo(**c) for c in chunks]
    except Exception as e:
        logger.error(f"获取文档切片错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{file_name:path}", summary="删除指定文档")
async def delete_document(file_name: str):
    """删除指定文档的所有切片。

    service 返回删除数量而非抛异常，**404 在这里判** ——
    HTTP 状态码是协议层的事，service 抛 HTTPException 会让
    CLI 调用方也得去 catch 一个 HTTP 异常。
    """
    try:
        deleted = await get_document_service().delete_document(file_name)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"文档 '{file_name}' 不存在")
        return {
            "message": "文档已删除",
            "file": file_name,
            "chunks_deleted": deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="获取系统统计")
async def get_stats():
    try:
        stats = await get_document_service().stats()
        return {
            **stats,
            "active_sessions": await get_chat_service().count_active_sessions(),
        }
    except Exception as e:
        logger.error(f"获取统计错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
