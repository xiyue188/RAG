"""
REST API路由（非流式）
"""

from fastapi import APIRouter, HTTPException, status
from backend.schemas import QueryRequest, QueryResponse, DocumentListResponse, CitationInfo
from backend.services.chat_service import get_chat_service
from rag.logger import get_logger
from rag import VectorDB

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat/message", response_model=QueryResponse, summary="发送对话消息（非流式）")
async def send_message(request: QueryRequest, session_id: str = None):
    """
    非流式对话接口，收集完整结果后返回。
    """
    try:
        chat_service = get_chat_service()
        session_id = chat_service.get_or_create_session(session_id)

        resolved_question = None
        answer_chunks = []
        citations = []

        async for chunk in chat_service.answer_stream(
            session_id=session_id,
            question=request.question,
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
        history = chat_service.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"获取历史错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/session/{session_id}", summary="清空会话")
async def clear_session(session_id: str):
    try:
        chat_service = get_chat_service()
        success = chat_service.clear_session(session_id)
        if success:
            return {"message": "会话已清空", "session_id": session_id}
        raise HTTPException(status_code=404, detail="会话不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空会话错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse, summary="获取文档列表")
async def list_documents():
    try:
        vectordb = VectorDB()
        collection = vectordb.get_collection()
        results = collection.get(include=["metadatas"])

        documents = []
        seen_files = set()
        for metadata in results.get("metadatas", []):
            file = metadata.get("file", "unknown")
            if file not in seen_files:
                documents.append({
                    "file": file,
                    "category": metadata.get("category", "unknown"),
                })
                seen_files.add(file)

        return DocumentListResponse(documents=documents, total=len(documents))

    except Exception as e:
        logger.error(f"获取文档列表错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="获取系统统计")
async def get_stats():
    try:
        vectordb = VectorDB()
        collection = vectordb.get_collection()
        count = collection.count()

        chat_service = get_chat_service()
        active_sessions = len(chat_service.sessions)

        return {
            "total_chunks": count,
            "active_sessions": active_sessions,
            "vectordb_name": collection.name
        }

    except Exception as e:
        logger.error(f"获取统计错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
