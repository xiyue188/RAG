"""
对话服务
六边形架构 - 适配器层，桥接rag/核心模块
"""

import uuid
import re
import asyncio
import queue
import threading
from collections import OrderedDict
from typing import Any, Dict, AsyncIterator, List, Optional
import time

from rag import Retriever, LLMClient, VectorDB, Embedder
from rag.conversation import ConversationManager, ReferenceResolver
from rag.llm import assess_context
from rag.logger import get_logger
# compute_relevance 仍用于给引用卡片补展示字段（检索层没给 relevance 时）。
# has_relevance_signal 不再在此处使用 —— 可答性判断已交回 assess_context。
from rag.scoring import compute_relevance

logger = get_logger(__name__)

# 内存里保留的最大会话数。内存只是热缓存，真相在 messages 表 ——
# 被淘汰的会话下次访问时从库里回填，不丢上下文
from backend.settings import MAX_SESSIONS  # noqa: E402

# 回填时从库里取的消息条数，与 ConversationManager 的 max_turns 对齐
_HISTORY_TURNS = 20


async def _persist_run_async(
    *,
    session_id: str,
    query: str,
    route: str,
    results: List[Dict],
    full_answer: str,
    citations: List[Dict],
    total_ms: int,
    first_token_ms: Optional[int],
) -> None:
    """一次 Q&A 结束后落库：session + run + evidence + 两条 message。

    失败只记日志，不抛异常 —— SSE 流不能因为落库问题中断。
    所有写入在同一个 session_scope() 事务里：要么全提交，要么全回滚。
    没有部分成功的状态。

    **调用时机**：在 `done` 事件 yield 之前。此时 full_answer 已完整累积，
    citations 已抽取，results 已固定。
    """
    # 延迟导入：避免模块加载时循环依赖（session_scope 导入 config，
    # config 在测试里可能比 chat_service 更早被 mock）
    from backend.db.session import session_scope
    from backend.repositories import (
        EvidenceRepository,
        RunRepository,
        SessionRepository,
    )

    try:
        async with session_scope() as db:
            session_repo = SessionRepository(db)
            run_repo = RunRepository(db)
            evidence_repo = EvidenceRepository(db)

            # 1. 确保 Session 行存在（并发安全的 get-or-create）
            await session_repo.get_or_create(session_id)

            # 2. 建 Run 行，拿到自增 id
            run = await run_repo.create(
                query=query,
                session_id=session_id,
                route=route,
            )

            # 3. 批量写证据。bulk_insert 要求 chunk_id 字段名，
            #    而检索层的结果用 'id'；在这里做一次字段映射，
            #    不改 EvidenceRepository 的接口契约。
            normed = [
                {
                    "chunk_id": r.get("id", ""),
                    "file": r.get("metadata", {}).get("file"),
                    "relevance": r.get("relevance"),
                    "retrieved_by": r.get("retrieved_by"),
                }
                for r in results
            ]
            evidence_rows = await evidence_repo.bulk_insert(run.id, normed)

            # 4. 把答案实际引用的 chunk 标记为 used_in_answer=True
            cited_ids = [c["chunk_id"] for c in citations if c.get("chunk_id")]
            if cited_ids:
                await evidence_repo.mark_used(run.id, cited_ids)

            # 5. 写用户消息与助手消息，顺带刷新 session.last_active_at
            await session_repo.append_message(
                session_id,
                role="user",
                content=query,
            )
            await session_repo.append_message(
                session_id,
                role="assistant",
                content=full_answer,
                run_id=run.id,
            )

            # 6. 收尾：把 run 标为 ok 并记录耗时
            await run_repo.finish(
                run.id,
                status="ok",
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            )

    except Exception:
        # 只记日志，不向上抛。落库失败不能让用户已收到的回答消失。
        logger.warning("run 落库失败（session=%s），对话不受影响", session_id, exc_info=True)


async def _load_history_async(session_id: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """从 messages 表读会话历史，按时间正序。

    数据库不可用时返回空列表并记日志：读不到历史只是丢了上下文，
    不该让整次问答失败。调用方据此退化为一个新会话。
    """
    from backend.db.session import session_scope
    from backend.repositories import SessionRepository

    try:
        async with session_scope() as db:
            rows = await SessionRepository(db).history(session_id, limit=limit)
            return [
                {
                    "role": m.role,
                    "content": m.content,
                    "metadata": {"run_id": m.run_id} if m.run_id else {},
                    "timestamp": m.created_at.isoformat(),
                }
                for m in rows
            ]
    except Exception:
        logger.warning("读取会话历史失败（session=%s），按新会话处理", session_id, exc_info=True)
        return []


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

    ## 会话的两份状态

    - **messages 表**：真相源。每次问答结束由 `_persist_run_async` 写入。
    - **`self.sessions`**：`ConversationManager` 的热缓存，指代消解与
      上下文拼接要用它。按 LRU 保留至多 `MAX_SESSIONS` 个，进程重启或被
      淘汰后，首次访问时从 messages 表回填。

    此前只有内存一份：重启即丢上下文，字典无上限，过期清理方法无人调用。
    """

    def __init__(self):
        logger.info("初始化ChatService...")
        self.vectordb = VectorDB()
        self.embedder = Embedder()
        self.llm = LLMClient()
        self.retriever = Retriever(self.vectordb, self.embedder, self.llm)

        self.sessions: "OrderedDict[str, ConversationManager]" = OrderedDict()
        self.resolvers: Dict[str, ReferenceResolver] = {}
        logger.info("ChatService初始化完成")

    @staticmethod
    def ensure_session_id(session_id: Optional[str] = None) -> str:
        """没给就生成一个。不创建内存状态 —— 那由 answer_stream 异步完成，
        因为可能要从数据库回填，而回填是 I/O，不能在同步方法里做。"""
        return session_id or str(uuid.uuid4())

    async def _get_conversation(self, session_id: str) -> ConversationManager:
        """取会话上下文：内存命中直接用，否则从 messages 表回填。"""
        conversation = self.sessions.get(session_id)
        if conversation is not None:
            self.sessions.move_to_end(session_id)
            return conversation

        history = await _load_history_async(session_id, limit=_HISTORY_TURNS)
        loaded = ConversationManager(max_turns=_HISTORY_TURNS)
        for turn in history:
            if turn["role"] == "user":
                loaded.add_user_message(turn["content"])
            else:
                loaded.add_assistant_message(turn["content"])

        # await 期间同一会话的另一个请求可能已经回填并开始写入，
        # 用 setdefault 保留先到的那份，不覆盖
        conversation = self.sessions.setdefault(session_id, loaded)
        self.sessions.move_to_end(session_id)
        if session_id not in self.resolvers:
            self.resolvers[session_id] = ReferenceResolver(conversation)
        if history:
            logger.info(f"[{session_id}] 从数据库回填 {len(history)} 条历史消息")

        while len(self.sessions) > MAX_SESSIONS:
            evicted, _ = self.sessions.popitem(last=False)
            self.resolvers.pop(evicted, None)
        return conversation

    async def answer_stream(
        self,
        session_id: str,
        question: str,
        use_retrieval: bool = True,
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

        session_id = self.ensure_session_id(session_id)
        conversation = await self._get_conversation(session_id)
        resolver = self.resolvers[session_id]

        yield {"type": "connected", "data": {"session_id": session_id}}

        # 计时与落库所需的上下文变量。
        # 放在 try 外面，这样即使 try 内部分支赋值，finally / 落库代码都能拿到。
        _t0: float = time.monotonic()
        _first_token_time: Optional[float] = None
        _route: str = "smart"   # citation | smart，在分支决策时更新
        citations: List[Dict] = []  # 引用模式下会被 _citations 内部事件填充

        try:
            # 1. 指代消解
            resolved_question = await asyncio.to_thread(resolver.resolve, question)
            yield {
                "type": "resolved",
                "data": {"original": question, "resolved": resolved_question}
            }

            # 1.5 对话感知查询增强（实体注入，无需 LLM，快速可靠）
            # 策略：追问通常很短且模糊，从对话历史中提取实体拼接到查询前
            # 示例："能查到原因吗" → "船员头晕 能查到原因吗" → 检索命中正确文档
            retrieval_query = resolved_question
            if use_retrieval and len(conversation.history) >= 2:
                q = resolved_question.strip()
                logger.info(f"[{session_id}] [查询增强] 历史={len(conversation.history)}条, 查询长度={len(q)}, 查询='{q}'")
                # 仅对短查询（≤20字）注入实体，长查询本身已有足够语义
                if len(q) <= 20:
                    entities = conversation.extract_entities()
                    logger.info(f"[{session_id}] [查询增强] 提取实体={entities}")
                    if entities:
                        entity_prefix = " ".join(entities[:3])
                        retrieval_query = f"{entity_prefix} {q}"
                        logger.info(f"[{session_id}] [查询增强] 实体注入: '{q}' → '{retrieval_query}'")
                        yield {
                            "type": "query_rewritten",
                            "data": {"original": resolved_question, "rewritten": retrieval_query}
                        }
                    else:
                        # 实体提取失败时回退：直接拼接上一轮用户问题作为上下文
                        last_user_q = conversation.get_last_user_message()
                        if last_user_q and last_user_q.strip() != q:
                            retrieval_query = f"{last_user_q} {q}"
                            logger.info(f"[{session_id}] [查询增强] 历史拼接(回退): '{q}' → '{retrieval_query}'")
                            yield {
                                "type": "query_rewritten",
                                "data": {"original": resolved_question, "rewritten": retrieval_query}
                            }

            # 2. 检索阶段（如果不使用知识库则跳过）
            results = []
            if use_retrieval:
                # 2. 高级检索（带详细日志事件）
                yield {"type": "retrieval_status", "data": {"status": "searching"}}
                logger.info(f"[{session_id}] 已发送 retrieval_status 事件")

                # 2.1 多查询扩展开始（hybrid 和 multi_query 互斥，hybrid 优先）
                if enable_multi_query and not enable_hybrid:
                    logger.info(f"[{session_id}] 准备发送 multi_query_start 事件")
                    yield {
                        "type": "multi_query_start",
                        "data": {"original": retrieval_query}
                    }
                    logger.info(f"[{session_id}] 已发送 multi_query_start 事件")

                advanced_result = await asyncio.to_thread(
                    self.retriever.retrieve_advanced,
                    retrieval_query,
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
                    # RRF 融合没有权重参数：只用排名，不用分数。
                    # 此前这里硬编码 0.7/0.3，即使配置改了前端显示也不会变。
                    from config import RRF_K
                    yield {
                        "type": "hybrid_search_start",
                        "data": {
                            "method": "hybrid",
                            "fusion": "rrf",
                            "rrf_k": RRF_K
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
                # relevance 由 rag.scoring 统一计算（[0,1] 越大越相关）。
                # 此前用 `or` 链在 rerank_score / hybrid_score / 1-distance
                # 之间回退，既混用不同物理量，又会把合法的 0.0 判为缺失。
                result_details = []
                for i, r in enumerate(results[:5], 1):  # 只推送前5个
                    relevance = r.get('relevance')
                    if not isinstance(relevance, (int, float)):
                        relevance = compute_relevance(r)

                    detail = {
                        "rank": i,
                        "file": r.get('metadata', {}).get('file', 'unknown'),
                        "category": r.get('metadata', {}).get('category', 'unknown'),
                        "relevance": round(float(relevance), 3),
                        # relevance 的物理含义取决于是否经过精排，前端据此标注口径：
                        #   rerank — cross-encoder 判定相关的概率 sigmoid(logit)
                        #   cosine — 归一化向量的余弦相似度 1 - d/2
                        "relevance_basis": (
                            "rerank" if isinstance(r.get('rerank_logit'), (int, float))
                            else "cosine"
                        ),
                        "retrieved_by": r.get('retrieved_by', []),
                    }

                    cosine_distance = r.get('cosine_distance')
                    if isinstance(cosine_distance, (int, float)):
                        detail["cosine_distance"] = round(float(cosine_distance), 3)
                    if isinstance(r.get('rrf_score'), (int, float)):
                        detail["rrf_score"] = round(float(r['rrf_score']), 5)

                    result_details.append(detail)

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
            full_prompt = ""  # 用于 Prompt Inspector

            # 可答性判断由 rag.llm.assess_context 负责 —— 那是唯一实现。
            #
            # 此前这里有一份手抄副本（relevance_of / top_relevance = max(...) /
            # has_signal = any(...) / 阈值比较，四部分与 assess_context 逐一
            # 对应），而下面的 answer_smart_stream 又会让 rag/ 再算一遍，
            # 同一判断一次请求执行两次。
            #
            # 判断"证据够不够"需要领域知识（relevance 口径、各检索方案的
            # 分数分布、M1 实测校准的阈值），属于 rag/ 而非编排层。
            # 编排层只消费结论：决定走引用模式还是智能模式。
            should_use_context, _, top_relevance, _ = assess_context(results)

            logger.info(
                f"[{session_id}] 可答性检查: top_relevance="
                f"{'None' if top_relevance is None else f'{top_relevance:.3f}'}, "
                f"use_context={should_use_context}"
            )

            # 如果有高质量检索结果且启用引用，使用引用模式
            if enable_citation and should_use_context:
                _route = "citation"
                async for event in self._stream_with_citations(
                    resolved_question, results, conversation_context
                ):
                    if event["type"] == "answer_chunk":
                        if _first_token_time is None:
                            _first_token_time = time.monotonic()
                        full_answer += event["data"]["content"]
                    elif event["type"] == "_citations":
                        citations = event["data"]
                        continue  # 内部事件，不向外发送
                    elif event["type"] == "_metadata":
                        # 捕获 prompt（内部事件）
                        full_prompt = event["data"].get("full_prompt", "")
                        logger.info(f"[{session_id}] [DEBUG] 收到_metadata事件，full_prompt长度: {len(full_prompt)}")
                        continue
                    yield event

                if citations:
                    yield {"type": "citations", "data": {"citations": citations}}

                if not full_answer.strip():
                    logger.warning(
                        f"[{session_id}] 引用流式生成完成但答案为空，切换到非流式补偿生成"
                    )
                    fallback_text = await asyncio.to_thread(
                        self.llm.generate,
                        full_prompt,
                    )
                    fallback_text = (fallback_text or "").strip()

                    if fallback_text:
                        if _first_token_time is None:
                            _first_token_time = time.monotonic()
                        full_answer = fallback_text
                        yield {"type": "answer_chunk", "data": {"content": fallback_text}}
                    else:
                        raise RuntimeError("LLM 生成完成，但返回内容为空，请检查当前模型配置。")
            else:
                # 使用智能模式（混合式RAG：有好结果用文档，无结果或差结果用通用知识）
                _route = "smart"
                logger.info(f"[{session_id}] 使用智能模式（混合式RAG）")
                # 不传 answerable_min：answer_smart_stream 内部同样调
                # assess_context，其默认值就是 config 里那个。编排层把阈值
                # 读出来再传进去，只是给"两处不一致"制造机会。
                async for chunk in _sync_generator_to_async(
                    lambda: self.llm.answer_smart_stream(
                        resolved_question,
                        results,
                        conversation_context=conversation_context
                    )
                ):
                    # 跳过元数据字典（但捕获 full_prompt）
                    if isinstance(chunk, dict):
                        metadata = chunk
                        full_prompt = metadata.get('full_prompt', '')
                        logger.info(f"[{session_id}] 智能模式: {metadata.get('mode')} - {metadata.get('reason')}")
                        continue

                    if _first_token_time is None:
                        _first_token_time = time.monotonic()
                    full_answer += chunk
                    yield {"type": "answer_chunk", "data": {"content": chunk}}

            # 4. 保存对话历史（统一处理，使用原始question）
            conversation.add_user_message(question)
            conversation.add_assistant_message(full_answer)
            logger.info(f"[{session_id}] 回答完成，长度: {len(full_answer)}")
            logger.info(f"[{session_id}] [DEBUG] 发送done事件，full_prompt长度: {len(full_prompt)}")

            # 5. 异步落库（session / run / evidence / messages）。
            #    失败只记日志，不中断 SSE 流 —— 用户已收到的回答不能因此丢失。
            total_ms = int((time.monotonic() - _t0) * 1000)
            first_token_ms = (
                int((_first_token_time - _t0) * 1000)
                if _first_token_time is not None
                else None
            )
            await _persist_run_async(
                session_id=session_id,
                query=question,
                route=_route,
                results=results,
                full_answer=full_answer,
                citations=citations,
                total_ms=total_ms,
                first_token_ms=first_token_ms,
            )

            yield {"type": "done", "data": {"success": True, "full_prompt": full_prompt}}

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"[{session_id}] 错误: {str(e)}\n{error_detail}")
            yield {
                "type": "error",
                "data": {
                    "message": str(e),
                    "error_type": type(e).__name__,
                    "detail": error_detail[:500]  # 限制长度
                }
            }

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
            _metadata: 内部事件，包含 full_prompt
        """
        # 🎯 使用实际的 chunk ID（来自向量数据库），而不是临时的 doc_1, doc_2
        doc_map = {}
        for i, r in enumerate(results):
            doc_num = f"doc_{i+1}"
            chunk_id = r.get('id', doc_num)
            doc_map[doc_num] = {
                'chunk_id': chunk_id,
                'file': r.get('metadata', {}).get('file', 'unknown'),
                'category': r.get('metadata', {}).get('category', 'unknown'),
                # 修复：检索结果用 'document' 字段存原文，不是 'content'
                'content': r.get('document', ''),
                # 引用卡片展示的相关性，与检索面板同一口径
                'relevance': (
                    r['relevance'] if isinstance(r.get('relevance'), (int, float))
                    else compute_relevance(r)
                ),
            }

        citations = []
        buffer = ""
        full_prompt = ""

        # 角标编号按 chunk_id 去重：同一 chunk 被引多次共享同一编号。
        # 此前每个 [doc_X] 都替换成 "[来源: 文件名]"，同一 chunk 引三次
        # 就出现三段冗长重复的文本，占版面且不传达新信息。
        chunk_to_number = {}
        # citations 与角标编号一一对应，每个编号只推一条
        emitted_numbers = set()

        def number_for(chunk_id: str) -> int:
            if chunk_id not in chunk_to_number:
                chunk_to_number[chunk_id] = len(chunk_to_number) + 1
            return chunk_to_number[chunk_id]

        async for chunk in _sync_generator_to_async(
            lambda: self.llm.answer_with_citations_stream(
                question, results, conversation_context
            )
        ):
            # 捕获元数据字典中的 full_prompt
            if isinstance(chunk, dict):
                prompt_value = chunk.get('full_prompt', '')
                if prompt_value:
                    full_prompt = prompt_value
                logger.info(f"[DEBUG] 捕获到元数据，full_prompt 长度: {len(prompt_value)} 字符 (已保存: {len(full_prompt)})")
                continue

            buffer += chunk

            # 尝试匹配 [doc_X] 标记
            match = re.search(r'\[doc_(\d+)\]', buffer)
            if match:
                doc_id = f"doc_{match.group(1)}"
                doc_info = doc_map.get(doc_id, {})
                pre_match = buffer[:match.start()]

                if doc_info:
                    chunk_id = doc_info.get('chunk_id', doc_id)
                    number = number_for(chunk_id)
                    # 只输出紧凑角标，不拼文件名 —— 呈现形式交给前端，
                    # 后端塞长文本会让展示层无法改样式
                    output_chunk = pre_match + f"[{number}]"

                    # 每个编号只推一条 citation，前端据 number 建立映射
                    if number not in emitted_numbers:
                        emitted_numbers.add(number)
                        citations.append({
                            "number": number,
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "file": doc_info.get('file', 'unknown'),
                            "category": doc_info.get('category', 'unknown'),
                            "content": doc_info.get('content', ''),
                            "relevance": doc_info.get('relevance', 0.0),
                        })
                else:
                    # 模型引用了不存在的 doc_X，丢弃该标记而非原样输出
                    output_chunk = pre_match

                yield {"type": "answer_chunk", "data": {"content": output_chunk}}
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

        # 发送内部 prompt 事件
        if full_prompt:
            yield {"type": "_metadata", "data": {"full_prompt": full_prompt}}

    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """完整会话历史，以数据库为准。重启后仍可读（验收 C5）。"""
        return await _load_history_async(session_id, limit=None)

    async def clear_session(self, session_id: str) -> bool:
        """删除会话：内存与数据库都清。返回是否存在过。

        数据库里 messages 随 sessions 级联删除；runs.session_id 是
        SET NULL —— 执行轨迹保留，可追溯性不随会话删除而丢失。
        """
        from backend.db.session import session_scope
        from backend.repositories import SessionRepository

        in_memory = self.sessions.pop(session_id, None) is not None
        self.resolvers.pop(session_id, None)
        async with session_scope() as db:
            in_db = await SessionRepository(db).clear(session_id)
        if in_memory or in_db:
            logger.info(f"清空会话: {session_id}")
        return in_memory or in_db

    async def count_active_sessions(self, within_seconds: int = 3600) -> int:
        """近期活跃会话数，以数据库为准；数据库不可用时退回内存计数。"""
        from backend.db.session import session_scope
        from backend.repositories import SessionRepository

        try:
            async with session_scope() as db:
                return await SessionRepository(db).count_active(within_seconds)
        except Exception:
            logger.warning("统计活跃会话失败，退回内存计数", exc_info=True)
            return len(self.sessions)


# 全局单例
_chat_service_instance: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """获取ChatService单例"""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
