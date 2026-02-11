"""
SSE (Server-Sent Events) 流式接口
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.services.chat_service import get_chat_service
from backend.schemas import QueryRequest
from rag.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _sse_event_generator(
    question: str,
    session_id: str = None,
    enable_multi_query: bool = True,
    enable_rerank: bool = False,
    enable_hybrid: bool = True,
    enable_citation: bool = True
):
    """SSE事件生成器（POST/GET共用）"""
    try:
        chat_service = get_chat_service()

        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())

        session_id = chat_service.get_or_create_session(session_id)
        logger.info(f"[SSE] 开始流式对话: {session_id}, 问题: {question}")

        async for chunk in chat_service.answer_stream(
            session_id=session_id,
            question=question,
            enable_multi_query=enable_multi_query,
            enable_rerank=enable_rerank,
            enable_hybrid=enable_hybrid,
            enable_citation=enable_citation
        ):
            event_data = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {event_data}\n\n"

        logger.info(f"[SSE] 流式对话完成: {session_id}")

    except Exception as e:
        logger.error(f"[SSE] 错误: {str(e)}", exc_info=True)
        error_data = json.dumps({
            "type": "error",
            "data": {"message": str(e)}
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"


@router.post("/chat/stream")
async def chat_stream_post(request: QueryRequest, session_id: str = None):
    """
    SSE流式对话接口（POST版本）

    适合前端使用 fetch + ReadableStream 调用。
    """
    return StreamingResponse(
        _sse_event_generator(
            question=request.question,
            session_id=session_id,
            enable_multi_query=request.enable_multi_query,
            enable_rerank=request.enable_rerank,
            enable_hybrid=request.enable_hybrid,
            enable_citation=request.enable_citation
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/chat/stream-get")
async def chat_stream_get(
    question: str,
    session_id: str = None,
    enable_multi_query: bool = True,
    enable_rerank: bool = False,
    enable_hybrid: bool = True,
    enable_citation: bool = True
):
    """
    SSE流式对话接口（GET版本）

    适合前端 EventSource API（只支持GET请求）。

    使用方法:
    ```javascript
    const url = '/api/v1/chat/stream-get?' + new URLSearchParams({
        question: '什么是宠物政策？',
        session_id: 'test123',
        enable_citation: 'true'
    });
    const eventSource = new EventSource(url);
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'answer_chunk') {
            document.getElementById('answer').innerText += data.data.content;
        } else if (data.type === 'done') {
            eventSource.close();
        }
    };
    ```
    """
    return StreamingResponse(
        _sse_event_generator(
            question=question,
            session_id=session_id,
            enable_multi_query=enable_multi_query,
            enable_rerank=enable_rerank,
            enable_hybrid=enable_hybrid,
            enable_citation=enable_citation
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
