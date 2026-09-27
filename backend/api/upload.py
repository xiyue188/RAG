"""
文档上传 API

两种上传模式：
- 同步：POST /documents/upload — 等摄入完成再返回，适合小文件
- 流式：POST /documents/upload/stream — SSE 进度流，适合前端实时反馈
- 异步：POST /documents/upload/async — 立即返回 task_id，后台处理（T6 新增）
  配合 GET /documents/tasks/{task_id} 查询状态
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
import json

# 文件大小限制：10MB（防止恶意大文件导致 OOM）
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

from config import SUPPORTED_FILE_TYPES
from backend.adapters import StreamingIngestionAdapter
from backend.services.document_service import (
    get_ingestion_async,
    get_document_service,
)

# 用标准库 logging：协议层不 import rag（见 tests/test_architecture.py）
logger = logging.getLogger(__name__)
router = APIRouter()
SUPPORTED_UPLOAD_TYPES = {ext.lower() for ext in SUPPORTED_FILE_TYPES}

def _write_temp_file(content: bytes, suffix: str) -> str:
    """把上传内容落到临时文件，返回路径。阻塞，需 to_thread 调用。"""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fp:
        fp.write(content)
        return fp.name


def _remove_temp_file(path: str) -> None:
    """删除临时文件。阻塞，需 to_thread 调用。"""
    Path(path).unlink(missing_ok=True)


async def invalidate_retrieval_caches() -> None:
    """文档变更后刷新检索侧缓存。

    收敛到 service 一处，两个上传端点共用 —— 此前两处各写了一份
    `get_chat_service().retriever.invalidate_bm25_cache()`，
    从协议层穿三层去拿另一个 service 的内部对象的内部方法。
    """
    await get_document_service().invalidate_retrieval_caches()


@router.post("/documents/upload", summary="上传文档（同步版本）")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    上传文档并摄入到向量数据库

    支持的文件格式：.md, .txt

    返回：
    - success: bool - 是否成功
    - files_processed: int - 处理的文件数
    - total_chunks: int - 生成的总块数
    - files: List[Dict] - 每个文件的详情
    """
    logger.info(f"收到上传请求：{len(files)} 个文件")

    ingestion = await get_ingestion_async()

    results = []
    total_chunks = 0
    success_count = 0

    for file in files:
        file_result = {
            "filename": file.filename,
            "size": 0,
            "chunks": 0,
            "success": False,
            "error": None
        }

        try:
            # 检查文件类型
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_UPLOAD_TYPES:
                file_result["error"] = f"不支持的文件格式：{file_ext}"
                logger.warning(f"跳过不支持的文件：{file.filename}")
                results.append(file_result)
                continue

            # 读取文件内容（先检查大小）
            content = await file.read()
            file_result["size"] = len(content)

            if len(content) > MAX_FILE_SIZE_BYTES:
                file_result["error"] = f"文件过大（{len(content) // 1024}KB），上限 10MB"
                logger.warning(f"拒绝过大文件：{file.filename} ({len(content)} bytes)")
                results.append(file_result)
                continue

            # 写临时文件是阻塞磁盘 I/O，扔进线程池
            temp_path = await asyncio.to_thread(_write_temp_file, content, file_ext)

            logger.info(f"开始摄入文件：{file.filename} ({len(content)} bytes)")

            # 摄入文件
            try:
                # 提取类别（从文件名或默认）
                category = "uploaded"

                # **必须 to_thread。** ingest_file 一口气做完解析、分块、
                # embedding、写向量库，全是同步调用。直接在 async def 里跑
                # 会把事件循环占满整个摄入时长 —— 10MB 文件期间所有请求
                # （包括 /health）都在等，而健康检查超时会让容器被重启。
                chunk_count = await asyncio.to_thread(
                    ingestion.ingest_file,
                    temp_path,
                    category=category,
                    original_filename=file.filename,
                )

                file_result["chunks"] = chunk_count
                file_result["success"] = True

                total_chunks += chunk_count
                success_count += 1

                logger.info(f"✓ 文件摄入成功：{file.filename} ({chunk_count} chunks)")

            finally:
                await asyncio.to_thread(_remove_temp_file, temp_path)

        except Exception as e:
            file_result["error"] = str(e)
            logger.error(f"✗ 文件摄入失败：{file.filename} - {e}", exc_info=True)

        results.append(file_result)

    # 返回结果
    response = {
        "success": success_count > 0,
        "files_processed": success_count,
        "files_failed": len(files) - success_count,
        "total_files": len(files),
        "total_chunks": total_chunks,
        "files": results
    }

    logger.info(
        f"上传完成：{success_count}/{len(files)} 文件成功，"
        f"生成 {total_chunks} 个 chunks"
    )

    # 刷新检索侧缓存（BM25 索引按全库语料构建，新增文档后需重建）
    if success_count > 0:
        await invalidate_retrieval_caches()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response
    )


@router.post("/documents/upload/stream", summary="上传文档（SSE 流式版本）")
async def upload_documents_stream(files: List[UploadFile] = File(...)):
    """
    上传文档并摄入到向量数据库（SSE 流式版本）

    支持的文件格式：.md, .txt

    返回 SSE 事件流，包含 12 种事件类型：
    1. file_received      - 文件接收
    2. parsing_start      - 开始解析
    3. parsing_done       - 解析完成
    4. chunking_start     - 开始分块
    5. chunking_done      - 分块完成
    6. embedding_start    - 开始 Embedding
    7. embedding_progress - Embedding 进度
    8. embedding_done     - Embedding 完成
    9. storing_start      - 开始存储
    10. storing_done      - 存储完成
    11. indexing_done     - 索引更新完成
    12. upload_complete   - 上传完成
    13. error             - 错误
    14. all_complete      - 所有文件处理完成
    """
    logger.info(f"收到 SSE 流式上传请求：{len(files)} 个文件")

    # 在生成器外把上传内容落到临时盘。
    #
    # 必须在生成器外做：`UploadFile` 的句柄在响应开始流式输出后就关了，
    # 生成器里再 read() 会失败。
    #
    # **但不能像原先那样把全部内容读进内存再放进列表** ——
    # 10 个 10MB 文件就是 100MB 常驻，且要等最后一个读完才开始处理。
    # 改为逐个落临时盘，内存里只留路径与大小。
    staged: List[dict] = []
    for file in files:
        content = await file.read()
        ext = Path(file.filename).suffix.lower()
        size = len(content)

        # 类型与大小在这里就判掉，不合格的不落盘 —— 免得为一个必然被
        # 拒绝的 10MB 文件白写一次磁盘
        if ext not in SUPPORTED_UPLOAD_TYPES or size > MAX_FILE_SIZE_BYTES:
            staged.append({
                "filename": file.filename, "ext": ext,
                "size": size, "temp_path": None,
            })
            continue

        temp_path = await asyncio.to_thread(_write_temp_file, content, ext)
        staged.append({
            "filename": file.filename, "ext": ext,
            "size": size, "temp_path": temp_path,
        })
        del content  # 及时释放，下一轮循环不叠加

    async def event_generator():
        """SSE 事件生成器"""
        total_files = len(staged)
        processed_count = 0
        failed_count = 0

        # 发送开始事件
        yield f"data: {json.dumps({'type': 'upload_start', 'data': {'total_files': total_files}})}\n\n"

        ingestion = await get_ingestion_async()
        adapter = StreamingIngestionAdapter(ingestion)

        for file_index, fd in enumerate(staged, 1):
            filename = fd["filename"]
            file_ext = fd["ext"]
            temp_path = fd["temp_path"]

            # 检查文件类型
            if file_ext not in SUPPORTED_UPLOAD_TYPES:
                logger.warning(f"跳过不支持的文件：{filename}")
                yield f"data: {json.dumps({'type': 'file_skipped', 'data': {'filename': filename, 'reason': f'不支持的文件格式：{file_ext}'}})}\n\n"
                failed_count += 1
                continue

            # 处理文件内容
            try:

                # 检查文件大小（落盘阶段已判过，这里是它的对外表现）
                size = fd["size"]
                if size > MAX_FILE_SIZE_BYTES:
                    logger.warning(f"拒绝过大文件：{filename} ({size} bytes)")
                    message = f"文件过大（{size // 1024}KB），上限 10MB"
                    yield f"data: {json.dumps({'type': 'error', 'data': {'filename': filename, 'stage': 'file_read', 'message': message}}, ensure_ascii=False)}\n\n"
                    failed_count += 1
                    continue

                logger.info(f"[{file_index}/{total_files}] 开始处理：{filename}")

                # 流式摄入
                try:
                    async for event in adapter.ingest_file_stream(
                        temp_path,
                        filename,
                        category="uploaded"
                    ):
                        # 将事件转换为 SSE 格式
                        event_json = json.dumps(event, ensure_ascii=False)
                        yield f"data: {event_json}\n\n"

                        # 检查是否完成或错误
                        if event["type"] == "upload_complete":
                            processed_count += 1
                        elif event["type"] == "error":
                            failed_count += 1

                finally:
                    await asyncio.to_thread(_remove_temp_file, temp_path)

            except Exception as e:
                logger.error(f"处理文件失败：{filename} - {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'data': {'filename': filename, 'stage': 'file_read', 'message': str(e)}}, ensure_ascii=False)}\n\n"
                failed_count += 1

        # 兜底清理：客户端中途断开时 SSE 生成器会被中止，上面每个文件的
        # finally 未必执行得到，已落盘但没处理的临时文件会残留。
        for fd in staged:
            if fd["temp_path"]:
                await asyncio.to_thread(_remove_temp_file, fd["temp_path"])

        # 刷新检索侧缓存（BM25 索引按全库语料构建，新增文档后需重建）
        if processed_count > 0:
            await invalidate_retrieval_caches()

        # 发送所有文件处理完成事件
        yield f"data: {json.dumps({'type': 'all_complete', 'data': {'total': total_files, 'success': processed_count, 'failed': failed_count}})}\n\n"
        logger.info(f"所有文件处理完成：{processed_count} 成功，{failed_count} 失败")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# T6：异步上传（立即返回 task_id，后台摄入）
# ============================================================

@router.post("/documents/upload/async", summary="异步上传文档（立即返回 task_id）")
async def upload_documents_async(files: List[UploadFile] = File(...)):
    """上传文档并在后台异步摄入。

    立即返回 `task_id` 列表，客户端通过
    `GET /api/v1/documents/tasks/{task_id}` 轮询状态。

    **适用场景**：大文件（5MB+）或批量上传时避免请求超时。

    返回格式：
    ```json
    {"tasks": [{"task_id": "...", "filename": "...", "status": "pending"|"error", "error": null}]}
    ```
    """
    from backend.db.session import session_scope
    from backend.repositories import IngestTaskRepository
    from backend.services import ingest_service

    results = []
    for file in files:
        file_ext = Path(file.filename).suffix.lower()

        # 类型校验
        if file_ext not in SUPPORTED_UPLOAD_TYPES:
            results.append({
                "task_id": None,
                "filename": file.filename,
                "status": "error",
                "error": f"不支持的文件格式：{file_ext}",
            })
            continue

        # 读取并校验大小
        content = await file.read()
        size = len(content)
        if size > MAX_FILE_SIZE_BYTES:
            results.append({
                "task_id": None,
                "filename": file.filename,
                "status": "error",
                "error": f"文件过大（{size // 1024}KB），上限 10MB",
            })
            continue

        # 写临时文件（阻塞 I/O 放线程池）
        task_id = str(uuid.uuid4())
        temp_path = await asyncio.to_thread(_write_temp_file, content, file_ext)
        del content  # 尽早释放内存

        # 在 PG 建任务行
        async with session_scope() as s:
            await IngestTaskRepository(s).create(
                filename=file.filename,
                size_bytes=size,
                category="uploaded",
                task_id=task_id,
            )

        # 入队，后台 worker 会取走执行
        await ingest_service.enqueue(task_id, temp_path, file.filename)

        results.append({
            "task_id": task_id,
            "filename": file.filename,
            "status": "pending",
            "error": None,
        })
        logger.info("async upload queued: %s → task_id=%s", file.filename, task_id)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"tasks": results},
    )


@router.get("/documents/tasks/{task_id}", summary="查询异步摄入任务状态")
async def get_ingest_task(task_id: str):
    """查询异步摄入任务的状态与进度。

    返回格式：
    ```json
    {
      "task_id": "...",
      "filename": "...",
      "status": "pending|running|done|error",
      "chunk_count": 42,
      "error": null,
      "progress": {"stage": "embedding", "progress": "42.0", "message": "..."}
    }
    ```

    `progress` 从 Redis 读取（高频更新），`status` 从 PG 读取（权威终态）。
    Redis 不可用时 `progress` 为空字典。
    """
    from backend.db.session import session_scope
    from backend.repositories import IngestTaskRepository
    from backend.cache import redis_client

    async with session_scope() as s:
        task = await IngestTaskRepository(s).get(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在",
        )

    # 读 Redis 进度（不可用时返回空字典，不报错）
    progress = await redis_client.hgetall(f"ingest:{task_id}")

    return {
        "task_id": task.id,
        "filename": task.filename,
        "size_bytes": task.size_bytes,
        "status": task.status,
        "chunk_count": task.chunk_count,
        "error": task.error,
        "created_at": task.created_at.isoformat(),
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        "progress": progress,
    }

