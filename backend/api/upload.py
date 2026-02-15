"""
文档上传 API
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
import tempfile
import shutil
from pathlib import Path
import json

from rag import DocumentIngestion, VectorDB, Embedder
from rag.logger import get_logger
from backend.adapters import StreamingIngestionAdapter

logger = get_logger(__name__)
router = APIRouter()

# 初始化摄入器（全局单例）
_ingestion_instance = None

def get_ingestion():
    """获取 DocumentIngestion 单例"""
    global _ingestion_instance
    if _ingestion_instance is None:
        logger.info("初始化 DocumentIngestion...")
        vectordb = VectorDB()
        embedder = Embedder()
        _ingestion_instance = DocumentIngestion(vectordb, embedder)
        logger.info("DocumentIngestion 初始化完成")
    return _ingestion_instance


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

    ingestion = get_ingestion()

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
            if file_ext not in [".md", ".txt"]:
                file_result["error"] = f"不支持的文件格式：{file_ext}"
                logger.warning(f"跳过不支持的文件：{file.filename}")
                results.append(file_result)
                continue

            # 读取文件内容
            content = await file.read()
            file_result["size"] = len(content)

            # 保存到临时文件
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=file_ext,
                delete=False
            ) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            logger.info(f"开始摄入文件：{file.filename} ({len(content)} bytes)")

            # 摄入文件
            try:
                # 提取类别（从文件名或默认）
                category = "uploaded"

                chunk_count = ingestion.ingest_file(
                    temp_path,
                    category=category
                )

                file_result["chunks"] = chunk_count
                file_result["success"] = True

                total_chunks += chunk_count
                success_count += 1

                logger.info(f"✓ 文件摄入成功：{file.filename} ({chunk_count} chunks)")

            finally:
                # 删除临时文件
                Path(temp_path).unlink(missing_ok=True)

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

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response
    )


@router.get("/documents", summary="获取文档列表")
async def list_documents():
    """获取已上传的文档列表"""
    try:
        ingestion = get_ingestion()
        collection = ingestion.vectordb.get_collection()
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

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "documents": documents,
                "total": len(documents)
            }
        )

    except Exception as e:
        logger.error(f"获取文档列表失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{filename}", summary="删除文档")
async def delete_document(filename: str):
    """删除指定文档的所有块"""
    try:
        ingestion = get_ingestion()
        collection = ingestion.vectordb.get_collection()

        # 查找该文件的所有块ID
        results = collection.get(
            where={"file": filename},
            include=["metadatas"]
        )

        ids_to_delete = results.get("ids", [])

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"文档不存在：{filename}"
            )

        # 删除
        collection.delete(ids=ids_to_delete)

        logger.info(f"已删除文档：{filename} ({len(ids_to_delete)} chunks)")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "filename": filename,
                "chunks_deleted": len(ids_to_delete)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Phase 2: SSE 流式上传 ==========

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

    async def event_generator():
        """SSE 事件生成器"""
        ingestion = get_ingestion()
        adapter = StreamingIngestionAdapter(ingestion)

        total_files = len(files)
        processed_count = 0
        failed_count = 0

        # 发送开始事件
        yield f"data: {json.dumps({'type': 'upload_start', 'data': {'total_files': total_files}})}\n\n"

        for file_index, file in enumerate(files, 1):
            # 检查文件类型
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in [".md", ".txt"]:
                logger.warning(f"跳过不支持的文件：{file.filename}")
                yield f"data: {json.dumps({'type': 'file_skipped', 'data': {'filename': file.filename, 'reason': f'不支持的文件格式：{file_ext}'}})}\n\n"
                failed_count += 1
                continue

            # 读取文件内容到临时文件
            try:
                content = await file.read()

                # 创建临时文件
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix=file_ext,
                    delete=False
                ) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name

                logger.info(f"[{file_index}/{total_files}] 开始处理：{file.filename}")

                # 流式摄入
                try:
                    async for event in adapter.ingest_file_stream(
                        temp_path,
                        file.filename,
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
                    # 删除临时文件
                    Path(temp_path).unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"处理文件失败：{file.filename} - {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'data': {'filename': file.filename, 'stage': 'file_read', 'message': str(e)}})}\n\n"
                failed_count += 1

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

