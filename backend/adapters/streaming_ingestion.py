"""
流式摄入适配器
将 DocumentIngestion 核心功能适配为 SSE 流式输出
不修改核心模块，在适配器层实现可观测性
"""

import sys
from pathlib import Path
from typing import Dict, AsyncIterator, Any
import asyncio

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag import DocumentIngestion
from rag.chunker import chunk_text
from rag.logger import get_logger
from config import ENCODING

logger = get_logger(__name__)


class StreamingIngestionAdapter:
    """
    流式摄入适配器

    职责：
    - 包装 DocumentIngestion 核心功能
    - 提供 SSE 流式进度输出
    - 不修改核心 RAG 模块
    - 遵循六边形架构原则

    事件类型（12种）：
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
    """

    def __init__(self, ingestion: DocumentIngestion):
        """
        初始化流式摄入适配器

        参数:
            ingestion: DocumentIngestion - 核心摄入器实例
        """
        self.ingestion = ingestion
        logger.info("StreamingIngestionAdapter 初始化完成")

    async def ingest_file_stream(
        self,
        file_path: str,
        filename: str,
        category: str = "uploaded"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式摄入单个文件

        通过调用核心模块的各个独立函数，在它们之间插入 SSE 事件。
        不修改核心模块，保持架构纯净。

        参数:
            file_path: str - 临时文件路径
            filename: str - 原始文件名
            category: str - 文档类别

        返回:
            AsyncIterator[Dict] - SSE 事件流

        异常:
            FileNotFoundError - 文件不存在
            UnicodeDecodeError - 文件编码错误
            Exception - 其他处理错误
        """
        file_path_obj = Path(file_path)

        try:
            # ========== 阶段 1: 文件接收 ==========
            file_size = file_path_obj.stat().st_size
            yield {
                "type": "file_received",
                "data": {
                    "filename": filename,
                    "size": file_size,
                    "category": category
                }
            }
            logger.info(f"[摄入] 文件接收: {filename} ({file_size} bytes)")

            # ========== 阶段 2: 文件解析 ==========
            yield {
                "type": "parsing_start",
                "data": {"filename": filename}
            }
            logger.info(f"[摄入] 开始解析: {filename}")

            # 读取文件内容
            await asyncio.sleep(0)  # 让出控制权，避免阻塞事件循环
            try:
                with open(file_path, 'r', encoding=ENCODING) as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                logger.error(f"[摄入] 编码错误: {filename} - {e}")
                yield {
                    "type": "error",
                    "data": {
                        "stage": "parsing",
                        "message": f"文件编码错误: {str(e)}"
                    }
                }
                return

            char_count = len(content)
            yield {
                "type": "parsing_done",
                "data": {
                    "filename": filename,
                    "chars": char_count
                }
            }
            logger.info(f"[摄入] 解析完成: {filename} ({char_count} 字符)")

            # ========== 阶段 3: 文本分块 ==========
            yield {
                "type": "chunking_start",
                "data": {"filename": filename}
            }
            logger.info(f"[摄入] 开始分块: {filename}")

            await asyncio.sleep(0)
            # 调用核心模块的分块函数
            chunks = chunk_text(content)
            chunk_count = len(chunks)

            yield {
                "type": "chunking_done",
                "data": {
                    "filename": filename,
                    "chunk_count": chunk_count
                }
            }
            logger.info(f"[摄入] 分块完成: {filename} ({chunk_count} chunks)")

            # ========== 阶段 4: 向量化 (Embedding) ==========
            yield {
                "type": "embedding_start",
                "data": {
                    "filename": filename,
                    "total_chunks": chunk_count
                }
            }
            logger.info(f"[摄入] 开始 Embedding: {filename} ({chunk_count} chunks)")

            # 批量向量化（核心模块方法）
            # 注意：实际的 encode 是批量操作，progress 是估算值
            await asyncio.sleep(0)

            # 分批处理，模拟进度（每批最多32个）
            batch_size = 32
            all_embeddings = []
            for i in range(0, chunk_count, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = self.ingestion.embedder.encode(
                    batch_chunks,
                    to_list=True
                )
                all_embeddings.extend(batch_embeddings)

                # 发送进度事件
                current = min(i + batch_size, chunk_count)
                yield {
                    "type": "embedding_progress",
                    "data": {
                        "filename": filename,
                        "current": current,
                        "total": chunk_count,
                        "percentage": round(current / chunk_count * 100, 1)
                    }
                }
                logger.debug(f"[摄入] Embedding 进度: {current}/{chunk_count}")

                await asyncio.sleep(0)  # 让出控制权

            embeddings = all_embeddings

            yield {
                "type": "embedding_done",
                "data": {
                    "filename": filename,
                    "chunk_count": chunk_count
                }
            }
            logger.info(f"[摄入] Embedding 完成: {filename}")

            # ========== 阶段 5: 存储到向量数据库 ==========
            yield {
                "type": "storing_start",
                "data": {"filename": filename}
            }
            logger.info(f"[摄入] 开始存储: {filename}")

            await asyncio.sleep(0)

            # 准备数据
            file_stem = Path(filename).stem
            ids = [f"{file_stem}_chunk_{i}" for i in range(chunk_count)]
            metadatas = [
                {
                    "file": filename,
                    "category": category
                }
                for _ in range(chunk_count)
            ]

            # 调用核心模块的存储方法
            self.ingestion.vectordb.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )

            yield {
                "type": "storing_done",
                "data": {
                    "filename": filename,
                    "chunk_count": chunk_count
                }
            }
            logger.info(f"[摄入] 存储完成: {filename}")

            # ========== 阶段 6: 索引更新 ==========
            await asyncio.sleep(0)
            total_docs = self.ingestion.vectordb.count()

            yield {
                "type": "indexing_done",
                "data": {
                    "filename": filename,
                    "total_docs_in_db": total_docs
                }
            }
            logger.info(f"[摄入] 索引更新完成: 数据库共 {total_docs} 个文档")

            # ========== 阶段 7: 上传完成 ==========
            yield {
                "type": "upload_complete",
                "data": {
                    "filename": filename,
                    "chunk_count": chunk_count,
                    "success": True
                }
            }
            logger.info(f"[摄入] 上传完成: {filename}")

        except FileNotFoundError as e:
            logger.error(f"[摄入] 文件不存在: {file_path} - {e}")
            yield {
                "type": "error",
                "data": {
                    "stage": "file_access",
                    "message": f"文件不存在: {filename}"
                }
            }

        except Exception as e:
            logger.error(f"[摄入] 处理失败: {filename} - {e}", exc_info=True)
            yield {
                "type": "error",
                "data": {
                    "stage": "unknown",
                    "message": f"处理失败: {str(e)}"
                }
            }

    def __repr__(self):
        return f"StreamingIngestionAdapter(ingestion={self.ingestion})"
