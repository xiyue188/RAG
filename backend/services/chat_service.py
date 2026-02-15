"""
对话服务
六边形架构 - 适配器层，桥接rag/核心模块
"""

import uuid
import re
import asyncio
import queue
import threading
from typing import Dict, AsyncIterator, List, Optional
from datetime import datetime

from rag import Retriever, LLMClient, VectorDB, Embedder
from rag.conversation import ConversationManager, ReferenceResolver
from rag.logger import get_logger

logger = get_logger(__name__)


async def _sync_generator_to_async(sync_gen_factory) -> AsyncIterator:
    """
    将同步生成器转换为异步迭代器。

    使用 Queue + Thread 模式：后台线程运行同步生成器，
    主协程通过 asyncio.to_thread 非阻塞地从队列消费。

    参数:
        sync_gen_factory: 无参函数，调用后返回同步生成器
    """
    chunk_queue = queue.Queue()

    def _producer():
        try:
            for item in sync_gen_factory():
                chunk_queue.put(('data', item))
            chunk_queue.put(('done', None))
        except Exception as e:
            chunk_queue.put(('error', e))

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        signal, value = await asyncio.to_thread(chunk_queue.get)
        if signal == 'done':
            break
        elif signal == 'error':
            raise value
        yield value


class ChatService:
    """
    对话服务（单例模式）

    职责：
    1. 管理多个用户会话（基于SessionID）
    2. 桥接rag/核心模块（不含业务逻辑）
    3. 提供异步流式输出接口
    """

    def __init__(self):
        logger.info("初始化ChatService...")
        self.vectordb = VectorDB()
        self.embedder = Embedder()
        self.llm = LLMClient()
        self.retriever = Retriever(self.vectordb, self.embedder, self.llm)

        self.sessions: Dict[str, ConversationManager] = {}
        self.resolvers: Dict[str, ReferenceResolver] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        logger.info("ChatService初始化完成")

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationManager(max_turns=20)
            self.resolvers[session_id] = ReferenceResolver(self.sessions[session_id])
            self.session_timestamps[session_id] = datetime.now()
            logger.info(f"初始化会话: {session_id}")

        self.session_timestamps[session_id] = datetime.now()
        return session_id

    async def answer_stream(
        self,
        session_id: str,
        question: str,
        enable_multi_query: bool = True,
        enable_rerank: bool = False,
        enable_hybrid: bool = True,
        enable_citation: bool = True
    ) -> AsyncIterator[Dict]:
        """
        流式回答（异步生成器）

        事件类型（完整 RAG 工作流）:
            1. connected - 会话已连接
            2. resolved - 问题消解（原始 → 解析后）
            3. retrieval_status - 开始检索
            4. multi_query_start - 多查询扩展开始
            5. multi_query_done - 多查询扩展完成（含变体）
            6. hybrid_search_start - 混合检索开始
            7. bm25_indexing - BM25 索引构建中
            8. bm25_indexed - BM25 索引完成
            9. rerank_start - Rerank 精排序开始
           10. rerank_done - Rerank 精排序完成
           11. retrieval_results - 检索结果详情（文件名+分数）
           12. retrieval_done - 检索完成
           13. generation_start - 开始生成答案
           14. answer_chunk - 答案文本片段（逐字）
           15. citations - 引用来源列表
           16. done - 回答完成
           17. error - 错误信息
        """
        logger.info(f"[{session_id}] 收到问题: {question}")

        session_id = self.get_or_create_session(session_id)
        conversation = self.sessions[session_id]
        resolver = self.resolvers[session_id]

        yield {"type": "connected", "data": {"session_id": session_id}}

        try:
            # 1. 指代消解
            resolved_question = await asyncio.to_thread(resolver.resolve, question)
            yield {
                "type": "resolved",
                "data": {"original": question, "resolved": resolved_question}
            }

            # 2. 高级检索（带详细日志事件）
            yield {"type": "retrieval_status", "data": {"status": "searching"}}
            logger.info(f"[{session_id}] ✓ 发送retrieval_status事件")

            # 2.1 多查询扩展开始
            if enable_multi_query:
                logger.info(f"[{session_id}] ✓ 准备发送multi_query_start事件")
                yield {
                    "type": "multi_query_start",
                    "data": {"original": resolved_question}
                }
                logger.info(f"[{session_id}] ✓ 已发送multi_query_start事件")

            advanced_result = await asyncio.to_thread(
                self.retriever.retrieve_advanced,
                resolved_question,
                enable_multi_query=enable_multi_query,
                enable_rerank=enable_rerank,
                enable_hybrid=enable_hybrid
            )
            results = advanced_result['results']
            stats = advanced_result.get('stats', {})
            expanded_queries = advanced_result.get('expanded_queries', [])

            # 2.2 多查询扩展完成
            if enable_multi_query and expanded_queries:
                yield {
                    "type": "multi_query_done",
                    "data": {
                        "original": resolved_question,
                        "variants": expanded_queries,
                        "count": len(expanded_queries)
                    }
                }

            # 2.3 检索方法通知
            retrieval_method = stats.get('retrieval_method', 'unknown')
            if retrieval_method == 'hybrid':
                yield {
                    "type": "hybrid_search_start",
                    "data": {
                        "method": "hybrid",
                        "vector_weight": 0.7,
                        "bm25_weight": 0.3
                    }
                }
                # BM25 索引构建（模拟）
                doc_count = self.vectordb.get_collection().count()
                yield {
                    "type": "bm25_indexing",
                    "data": {"doc_count": doc_count}
                }
                yield {
                    "type": "bm25_indexed",
                    "data": {"doc_count": doc_count}
                }

            # 2.4 Rerank 精排序
            if enable_rerank and stats.get('rerank_candidates', 0) > 0:
                yield {
                    "type": "rerank_start",
                    "data": {
                        "model": "BAAI/bge-reranker-base",
                        "candidates": stats['rerank_candidates']
                    }
                }
                yield {
                    "type": "rerank_done",
                    "data": {
                        "candidates": stats['rerank_candidates'],
                        "top_k": len(results)
                    }
                }

            # 2.5 检索结果详情
            result_details = []
            for i, r in enumerate(results[:5], 1):  # 只推送前5个
                result_details.append({
                    "rank": i,
                    "file": r.get('metadata', {}).get('file', 'unknown'),
                    "category": r.get('metadata', {}).get('category', 'unknown'),
                    "score": round(r.get('score', 0.0), 3),
                    "distance": round(r.get('distance', 0.0), 3)
                })

            yield {
                "type": "retrieval_results",
                "data": {
                    "results": result_details,
                    "total": len(results),
                    "method": retrieval_method
                }
            }

            logger.info(f"[{session_id}] 检索完成，找到 {len(results)} 个文档")
            yield {"type": "retrieval_done", "data": {"num_documents": len(results)}}

            # 3. 流式生成答案
            yield {"type": "generation_start", "data": {"num_docs": len(results)}}

            conversation_context = conversation.get_context_for_llm(max_turns=4)
            full_answer = ""
            citations = []

            # 智能判断：如果没有检索结果或相似度不足，使用混合式RAG（回退到通用知识）
            from config import SIMILARITY_THRESHOLD
            should_use_context = False

            if results:
                # 确保 distance 是 float 类型
                distance_value = results[0].get('distance', float('inf'))
                try:
                    max_similarity = float(distance_value)
                except (TypeError, ValueError):
                    max_similarity = float('inf')

                if max_similarity < float(SIMILARITY_THRESHOLD):
                    should_use_context = True
                logger.info(f"[{session_id}] 相似度检查: max={max_similarity:.3f}, threshold={SIMILARITY_THRESHOLD}, use_context={should_use_context}")

            # 如果有高质量检索结果且启用引用，使用引用模式
            if enable_citation and should_use_context:
                async for event in self._stream_with_citations(
                    resolved_question, results, conversation_context
                ):
                    if event["type"] == "answer_chunk":
                        full_answer += event["data"]["content"]
                    elif event["type"] == "_citations":
                        citations = event["data"]
                        continue  # 内部事件，不向外发送
                    yield event

                if citations:
                    yield {"type": "citations", "data": {"citations": citations}}
            else:
                # 使用智能模式（混合式RAG：有好结果用文档，无结果或差结果用通用知识）
                logger.info(f"[{session_id}] 使用智能模式（混合式RAG）")
                async for chunk in _sync_generator_to_async(
                    lambda: self.llm.answer_smart_stream(
                        resolved_question,
                        results,
                        threshold=float(SIMILARITY_THRESHOLD),  # 显式传入 float 类型
                        conversation_context=conversation_context
                    )
                ):
                    # 跳过元数据字典
                    if isinstance(chunk, dict):
                        metadata = chunk
                        logger.info(f"[{session_id}] 智能模式: {metadata.get('mode')} - {metadata.get('reason')}")
                        continue

                    full_answer += chunk
                    yield {"type": "answer_chunk", "data": {"content": chunk}}

            # 4. 保存对话历史（统一处理，使用原始question）
            conversation.add_user_message(question)
            conversation.add_assistant_message(full_answer)
            logger.info(f"[{session_id}] 回答完成，长度: {len(full_answer)}")

            yield {"type": "done", "data": {"success": True}}

        except Exception as e:
            logger.error(f"[{session_id}] 错误: {str(e)}", exc_info=True)
            yield {"type": "error", "data": {"message": str(e)}}

    async def _stream_with_citations(
        self,
        question: str,
        results: List[Dict],
        conversation_context: str
    ) -> AsyncIterator[Dict]:
        """
        引用模式的流式输出 + 实时替换 [doc_X] 标记。

        生成事件:
            answer_chunk: 替换后的文本片段
            _citations: 内部事件，收集到的引用列表
        """
        doc_map = {
            f"doc_{i+1}": {
                'file': r.get('metadata', {}).get('file', 'unknown'),
                'category': r.get('metadata', {}).get('category', 'unknown'),
                'content': r.get('content', ''),
                'score': r.get('score', 0.0)
            }
            for i, r in enumerate(results)
        }

        citations = []
        buffer = ""

        async for chunk in _sync_generator_to_async(
            lambda: self.llm.answer_with_citations_stream(
                question, results, conversation_context
            )
        ):
            # 跳过元数据字典
            if isinstance(chunk, dict):
                continue

            buffer += chunk

            # 尝试匹配 [doc_X] 标记
            match = re.search(r'\[doc_(\d+)\]', buffer)
            if match:
                doc_id = f"doc_{match.group(1)}"
                doc_info = doc_map.get(doc_id, {})
                file_name = doc_info.get('file', 'unknown')

                pre_match = buffer[:match.start()]
                replacement = f" [来源: {file_name}]"
                output_chunk = pre_match + replacement

                yield {"type": "answer_chunk", "data": {"content": output_chunk}}

                # 记录引用（去重）
                if doc_info and not any(c['doc_id'] == doc_id for c in citations):
                    citations.append({
                        "doc_id": doc_id,
                        "file": doc_info.get('file', 'unknown'),
                        "category": doc_info.get('category', 'unknown'),
                        "content": doc_info.get('content', ''),
                        "score": doc_info.get('score', 0.0)
                    })

                buffer = buffer[match.end():]

            elif len(buffer) > 20 and '[' not in buffer[-10:]:
                output = buffer[:-10]
                yield {"type": "answer_chunk", "data": {"content": output}}
                buffer = buffer[-10:]

        # 输出剩余buffer
        if buffer:
            yield {"type": "answer_chunk", "data": {"content": buffer}}

        # 发送内部引用事件
        if citations:
            yield {"type": "_citations", "data": citations}

    def get_session_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id].history

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.resolvers[session_id]
            del self.session_timestamps[session_id]
            logger.info(f"清空会话: {session_id}")
            return True
        return False

    def cleanup_old_sessions(self, timeout_seconds: int = 3600):
        now = datetime.now()
        expired = [
            sid for sid, ts in self.session_timestamps.items()
            if (now - ts).total_seconds() > timeout_seconds
        ]
        for sid in expired:
            self.clear_session(sid)
        if expired:
            logger.info(f"清理了 {len(expired)} 个超时会话")


# 全局单例
_chat_service_instance: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """获取ChatService单例"""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
