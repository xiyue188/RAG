"""文档管理服务

把 `routes.py` 里的文档列举/取切片/删除/统计搬到这里。搬的理由不是
"分层好看"，而是三条具体的问题：

1. **领域数据的变换发生在协议层。** `routes.py` 里有 60 行"按文件分组
   chunk"的逻辑 —— 换成 CLI 一字不变，它不是 HTTP 的事。想写个
   `scripts/list_documents.py` 就得抄一遍。
2. **每个请求新建一次 `VectorDB()`。** ChromaDB 客户端该是进程级单例，
   每请求新建等于重复初始化 —— 与 `db/session.py` 里反复强调的
   "engine 必须单例"是同一个错误，只是这次发生在向量库上。
3. **全部是同步阻塞调用跑在 `async def` 里。** 其中列举文档那条会拉全库
   正文，随语料线性增长；期间整个进程对所有请求失去响应，
   包括 Docker 的健康检查。

因此本模块的每个方法都 `await asyncio.to_thread(...)`：ChromaDB 的
Python 客户端是同步的，扔进线程池才不会占住事件循环。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from rag import DocumentIngestion, Embedder, VectorDB
from rag.logger import get_logger

logger = get_logger(__name__)

_vectordb: Optional[VectorDB] = None


def get_vectordb() -> VectorDB:
    """进程级单例。

    ChromaDB 客户端持有到磁盘的连接与内存索引，每请求新建会重复初始化。
    这与连接池必须复用是同一条理由。
    """
    global _vectordb
    if _vectordb is None:
        _vectordb = VectorDB()
    return _vectordb


def group_chunks_by_file(
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    *,
    include_chunks: bool,
) -> List[Dict[str, Any]]:
    """把扁平的 chunk 列表按来源文件聚成文档列表。

    **纯函数，不碰 I/O** —— 因此可以直接单测，不需要起 ChromaDB。
    这正是把它从 HTTP 端点里搬出来的收益：原先它嵌在 `routes.py` 的
    `try` 块里，只能靠起服务发请求才能验证。

    `include_chunks=False` 时不返回正文，只给计数。两个分支在原先的
    `routes.py` 里是两段几乎重复的代码，这里合并成一处 ——
    重复的分组逻辑是漂移的温床。

    参数:
        ids / documents / metadatas: ChromaDB `get()` 的三个平行列表。
            `documents` 允许为空列表（`include=["metadatas"]` 时）。
        include_chunks: 是否带上每个切片的正文。

    返回:
        [{file, category, chunks, chunk_count}, ...]
    """
    # documents 可能为空（只取 metadata 时），用 None 补齐以便 zip
    contents: List[Optional[str]] = list(documents) if documents else [None] * len(ids)

    grouped: Dict[str, Dict[str, Any]] = {}
    for chunk_id, content, metadata in zip(ids, contents, metadatas):
        metadata = metadata or {}
        file = metadata.get("file", "unknown")

        if file not in grouped:
            grouped[file] = {
                "file": file,
                "category": metadata.get("category", "unknown"),
                "chunks": [] if include_chunks else None,
                "chunk_count": 0,
            }

        entry = grouped[file]
        entry["chunk_count"] += 1
        if include_chunks:
            entry["chunks"].append({
                "id": chunk_id,
                "content": content or "",
                # 序号是"该文件内的第几块"，按加入顺序定 ——
                # 不用 metadata 里的 seq：那是摄入时写的，
                # 而这里要的是当前这批结果内的位置
                "index": len(entry["chunks"]),
            })

    return list(grouped.values())


class DocumentService:
    """文档的读写。所有 ChromaDB 调用都 offload 到线程池。"""

    def __init__(self, vectordb: Optional[VectorDB] = None) -> None:
        # 允许注入，便于测试；默认用单例
        self._vectordb = vectordb or get_vectordb()

    async def list_documents(self, include_chunks: bool = False) -> List[Dict[str, Any]]:
        """列出全部文档。

        `include_chunks=True` 会拉全库正文，代价随语料线性增长 ——
        默认关闭，前端按需再调 `get_chunks`。
        """
        include = ["metadatas", "documents"] if include_chunks else ["metadatas"]

        def _fetch() -> Dict[str, Any]:
            collection = self._vectordb.get_collection()
            return collection.get(include=include)

        results = await asyncio.to_thread(_fetch)
        return group_chunks_by_file(
            results.get("ids") or [],
            results.get("documents") or [],
            results.get("metadatas") or [],
            include_chunks=include_chunks,
        )

    async def get_chunks(self, file_name: str) -> List[Dict[str, Any]]:
        """取指定文档的全部切片，按加入顺序编号。"""

        def _fetch() -> Dict[str, Any]:
            collection = self._vectordb.get_collection()
            return collection.get(
                where={"file": file_name},
                include=["documents", "metadatas"],
            )

        results = await asyncio.to_thread(_fetch)
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        return [
            {"id": chunk_id, "content": content or "", "index": index}
            for index, (chunk_id, content) in enumerate(zip(ids, documents))
        ]

    async def delete_document(self, file_name: str) -> int:
        """删除文档的全部切片，返回删除数量。0 表示文档不存在。

        返回数量而不是抛 404：**"不存在"是调用方要判断的业务情况**，
        而 HTTP 状态码是协议层的事。Service 抛 HTTPException 就把协议
        细节泄漏进来了，CLI 调用方还得去 catch 一个 HTTP 异常。
        """

        def _delete() -> int:
            collection = self._vectordb.get_collection()
            existing = collection.get(where={"file": file_name}, include=["metadatas"])
            chunk_ids = existing.get("ids") or []
            if not chunk_ids:
                return 0
            collection.delete(ids=chunk_ids)
            return len(chunk_ids)

        deleted = await asyncio.to_thread(_delete)
        if deleted:
            logger.info("已删除文档 %s，共 %d 个切片", file_name, deleted)
            await self.invalidate_retrieval_caches()
        return deleted

    async def stats(self) -> Dict[str, Any]:
        """向量库统计。"""

        def _stats() -> Dict[str, Any]:
            collection = self._vectordb.get_collection()
            return {"total_chunks": collection.count(), "vectordb_name": collection.name}

        return await asyncio.to_thread(_stats)

    async def invalidate_retrieval_caches(self) -> None:
        """文档变更后让检索侧的缓存失效。

        BM25 索引是按全库语料构建的，删了文档不刷新会继续召回已删内容。

        原先 `routes.py` 写的是
        `get_chat_service().retriever.invalidate_bm25_cache()` ——
        从协议层穿三层去拿另一个 service 的内部对象的内部方法，
        这条链上任何一环改名都会断，而且 api 层本不该知道有 BM25 缓存。
        收敛到这里一处，调用方只需要知道"文档变了要通知一声"。

        同时自增 Redis 的 `kb_version`。目前还没有检索缓存读它（缓存已移出
        M2，见 STATUS），但"文档变了"的信号必须从唯一入口发出 ——
        同步上传、流式上传、异步上传、删除四条路径都经过这里。此前只有
        异步上传自增，将来接入缓存时另外三条路径会返回旧结果。

        失败不抛：缓存刷新失败的后果是检索结果略陈旧，
        而让删除操作整体失败的后果严重得多。`bump_kb_version` 本身在
        Redis 不可用时返回 None 不抛异常。
        """
        from backend.cache import redis_client

        await redis_client.bump_kb_version()
        try:
            from backend.services.chat_service import get_chat_service

            await asyncio.to_thread(
                get_chat_service().retriever.invalidate_bm25_cache
            )
        except Exception:  # noqa: BLE001 - 缓存刷新失败不该让业务失败
            logger.debug("刷新检索缓存失败，忽略并继续", exc_info=True)


_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """单例。与 `get_chat_service()` 一致的形状。"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


# ============================================================
# 摄入器
# ============================================================

_ingestion: Optional[DocumentIngestion] = None


def get_ingestion() -> DocumentIngestion:
    """摄入器单例。

    **首次调用会加载 embedding 模型，是秒级阻塞操作** ——
    异步调用方必须走 `get_ingestion_async()`。
    """
    global _ingestion
    if _ingestion is None:
        logger.info("初始化 DocumentIngestion...")
        _ingestion = DocumentIngestion(get_vectordb(), Embedder())
        logger.info("DocumentIngestion 初始化完成")
    return _ingestion


async def get_ingestion_async() -> DocumentIngestion:
    """异步入口。首次加载模型不阻塞事件循环。"""
    return await asyncio.to_thread(get_ingestion)
