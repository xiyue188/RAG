"""
检索模块
负责语义检索逻辑
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .vectordb import VectorDB
from .embedder import Embedder
from config import (
    TOP_K_RESULTS,
    RETRIEVAL_DISTANCE_THRESHOLD,
    ENABLE_THRESHOLD_FILTERING,
    ENABLE_AUTO_CLASSIFICATION,
    CATEGORY_KEYWORDS,
    RETRIEVAL_MODE,
    ENABLE_QUERY_REWRITE,
    ENABLE_MULTI_QUERY,
    NUM_EXPANDED_QUERIES,
    QUERY_REWRITE_PROMPT,
    MULTI_QUERY_PROMPT
)
from typing import List, Dict, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .llm import LLMClient 

logger = logging.getLogger(__name__)


class Retriever:
    """
    检索器
    封装语义检索逻辑
    """

    def __init__(self, vectordb: Optional[VectorDB] = None,
                 embedder: Optional[Embedder] = None,
                 llm: Optional["LLMClient"] = None):
        """
        初始化检索器

        参数:
            vectordb: VectorDB - 向量数据库实例（可选，默认创建新实例）
            embedder: Embedder - 向量化器实例（可选，默认创建新实例）
            llm: LLMClient - LLM客户端实例（可选，用于阶段2高级检索功能）
        """
        self.vectordb = vectordb or VectorDB()
        self.embedder = embedder or Embedder()
        self.llm = llm  # 可选，用于 Query Rewrite 和 Multi-Query

        # 确保集合已加载
        self.vectordb.get_collection()

    def retrieve(self, query: str, top_k: int = None,
                 category_filter: Optional[str] = None) -> List[Dict]:
        """
        检索相关文档

        参数:
            query: str - 用户查询
            top_k: int - 返回结果数量
            category_filter: str - 类别过滤（可选）

        返回:
            List[Dict] - 检索结果列表，每个结果包含:
                - id: 文档ID
                - document: 文档内容
                - metadata: 元数据
                - distance: 相似度距离（可选）
        """
        if top_k is None:
            top_k = TOP_K_RESULTS

        # 1. 将查询转换为向量
        query_embedding = self.embedder.encode(query, to_list=True)

        # 2. 构建过滤条件
        where = {"category": category_filter} if category_filter else None

        # 3. 查询向量数据库
        results = self.vectordb.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # 4. 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
            }
            # 添加距离信息（如果有）
            if 'distances' in results and results['distances']:
                result['distance'] = results['distances'][0][i]

            formatted_results.append(result)

        return formatted_results

    def retrieve_with_context(self, query: str, top_k: int = None) -> str:
        """
        检索相关文档并组合为上下文字符串

        参数:
            query: str - 用户查询
            top_k: int - 返回结果数量

        返回:
            str - 组合的上下文字符串
        """
        results = self.retrieve(query, top_k)

        # 组合上下文
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            doc = result['document']

            context_parts.append(
                f"[文档 {i}] 来源: {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}\n"
                f"内容: {doc}\n"
            )

        return "\n".join(context_parts)

    def _classify_query_by_keywords(self, query: str) -> Optional[str]:
        """
        基于关键词自动分类查询

        参数:
            query: str - 用户查询

        返回:
            str | None - 类别名称或 None（不过滤）
        """
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                logger.info(f"查询自动分类: '{query}' -> {category}")
                return category

        logger.info(f"查询未匹配任何类别: '{query}'")
        return None  # 不过滤

    def _extract_category_from_query(self, query: str) -> Optional[str]:
        """
        从查询中提取可能的类别名（metadata_only 模式）

        策略：检查查询中是否包含数据库已有的类别名

        参数:
            query: str - 用户查询

        返回:
            str | None - 类别名称或 None
        """
        try:
            # 获取数据库中所有已存在的类别（采样）
            sample = self.vectordb.get(limit=100)
            if not sample['metadatas']:
                return None

            existing_categories = set(
                m.get('category') for m in sample['metadatas']
                if m.get('category')
            )

            # 检查查询中是否包含类别名
            query_lower = query.lower()
            for category in existing_categories:
                if category.lower() in query_lower:
                    logger.info(f"查询包含类别名: '{query}' -> {category}")
                    return category

            return None
        except Exception as e:
            logger.warning(f"提取类别失败: {e}")
            return None

    def retrieve_with_threshold(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        带分数阈值过滤的检索

        参数:
            query: str - 用户查询
            top_k: int - 返回结果数量（默认使用 TOP_K_RESULTS）
            threshold: float - 距离阈值（默认使用 RETRIEVAL_DISTANCE_THRESHOLD）
            category_filter: str - 类别过滤（可选）

        返回:
            List[Dict] - 过滤后的检索结果列表
        """
        # 使用默认值
        if top_k is None:
            top_k = TOP_K_RESULTS
        if threshold is None:
            threshold = RETRIEVAL_DISTANCE_THRESHOLD

        # 1. 检索更多候选（2倍）以确保过滤后仍有足够结果
        candidate_k = top_k * 2
        results = self.retrieve(query, top_k=candidate_k, category_filter=category_filter)

        # 2. 过滤低质量结果
        filtered = [r for r in results if r.get('distance', 999) < threshold]

        logger.info(f"阈值过滤: {len(results)} -> {len(filtered)} (threshold={threshold})")

        # 3. 如果过滤后太少，放宽阈值
        if len(filtered) < 2 and len(results) > 0:
            relaxed_threshold = threshold * 1.5
            filtered = [r for r in results if r.get('distance', 999) < relaxed_threshold]
            logger.warning(
                f"过滤后结果不足，放宽阈值: {threshold} -> {relaxed_threshold}, "
                f"结果数: {len(filtered)}"
            )

        # 4. 返回 top-k
        return filtered[:top_k]

    def retrieve_smart(
        self,
        query: str,
        top_k: int = None,
        auto_classify: bool = None,
        enable_threshold: bool = None,
        mode: Optional[str] = None
    ) -> List[Dict]:
        """
        智能检索（支持三种模式）

        参数:
            query: str - 用户查询
            top_k: int - 返回结果数量
            auto_classify: bool - 是否自动分类（已弃用，请使用 mode 参数）
            enable_threshold: bool - 是否启用阈值过滤（默认使用 ENABLE_THRESHOLD_FILTERING）
            mode: str - 检索模式（可选）
                - "universal": 不分类，全库搜索
                - "metadata_only": 从查询中提取类别名（基于文件夹）
                - "keyword": 使用 CATEGORY_KEYWORDS 分类
                - None: 使用配置文件中的 RETRIEVAL_MODE

        返回:
            List[Dict] - 检索结果列表
        """
        # 确定检索模式
        if mode is None:
            mode = RETRIEVAL_MODE

        # 兼容旧参数：如果显式传了 auto_classify=False，覆盖为 universal 模式
        if auto_classify is False:
            mode = "universal"

        # 使用配置的默认值
        if enable_threshold is None:
            enable_threshold = ENABLE_THRESHOLD_FILTERING

        # 根据模式确定类别过滤
        category = None

        if mode == "keyword":
            # 关键词分类模式
            category = self._classify_query_by_keywords(query)
            logger.info(f"使用关键词模式，分类结果: {category}")

        elif mode == "metadata_only":
            # 元数据模式：从查询中提取类别名
            category = self._extract_category_from_query(query)
            logger.info(f"使用元数据模式，提取类别: {category}")

        elif mode == "universal":
            # 通用模式：不分类
            category = None
            logger.info("使用通用模式，不进行分类")

        else:
            logger.warning(f"未知检索模式: {mode}，使用通用模式")
            category = None

        # 执行检索
        if enable_threshold:
            results = self.retrieve_with_threshold(query, top_k, category_filter=category)
        else:
            results = self.retrieve(query, top_k, category_filter=category)

        return results

    # ============================================================
    # 阶段2: LLM增强检索方法
    # ============================================================

    def _rewrite_query(self, query: str) -> str:
        """
        使用 LLM 重写查询以优化检索效果

        参数:
            query: str - 原始用户查询

        返回:
            str - 重写后的查询（如果 LLM 不可用则返回原查询）
        """
        if self.llm is None:
            logger.warning("LLM 未配置，跳过查询重写")
            return query

        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            rewritten = self.llm.generate(prompt, max_tokens=100)
            rewritten = rewritten.strip()

            if rewritten:
                logger.info(f"查询重写: '{query}' -> '{rewritten}'")
                return rewritten
            else:
                logger.warning("LLM 返回空结果，使用原查询")
                return query

        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query

    def _expand_queries(self, query: str, n: int = None) -> List[str]:
        """
        使用 LLM 生成多个查询变体

        参数:
            query: str - 原始查询
            n: int - 生成的变体数量（默认使用 NUM_EXPANDED_QUERIES）

        返回:
            List[str] - 查询变体列表（包含原始查询）
        """
        if n is None:
            n = NUM_EXPANDED_QUERIES

        if self.llm is None:
            logger.warning("LLM 未配置，跳过多查询扩展")
            return [query]

        try:
            prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
            result = self.llm.generate(prompt, max_tokens=200)

            # 解析结果：每行一个查询
            variants = [line.strip() for line in result.strip().split('\n') if line.strip()]

            # 去重并限制数量
            variants = list(dict.fromkeys(variants))[:n]

            # 确保原始查询也在列表中
            if query not in variants:
                variants.insert(0, query)

            logger.info(f"多查询扩展: '{query}' -> {len(variants)} 个变体")
            return variants

        except Exception as e:
            logger.error(f"多查询扩展失败: {e}")
            return [query]

    def _retrieve_multi_query(
        self,
        queries: List[str],
        top_k: int = None,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        多查询检索并合并去重结果

        参数:
            queries: List[str] - 查询列表
            top_k: int - 每个查询返回的结果数量
            category_filter: str - 类别过滤（可选）

        返回:
            List[Dict] - 合并去重后的检索结果
        """
        if top_k is None:
            top_k = TOP_K_RESULTS

        all_results = {}  # 用 id 作为 key 去重

        for query in queries:
            results = self.retrieve(query, top_k=top_k, category_filter=category_filter)

            for result in results:
                doc_id = result['id']

                # 如果已存在，保留距离更小的结果
                if doc_id in all_results:
                    existing_distance = all_results[doc_id].get('distance', 999)
                    new_distance = result.get('distance', 999)
                    if new_distance < existing_distance:
                        all_results[doc_id] = result
                else:
                    all_results[doc_id] = result

        # 按距离排序
        merged = sorted(all_results.values(), key=lambda x: x.get('distance', 999))

        logger.info(f"多查询合并: {len(queries)} 个查询 -> {len(merged)} 个去重结果")

        return merged[:top_k * 2]  # 返回更多结果供后续过滤

    def retrieve_advanced(
        self,
        query: str,
        top_k: int = None,
        enable_rewrite: bool = None,
        enable_multi_query: bool = None,
        enable_threshold: bool = None,
        mode: Optional[str] = None
    ) -> Dict:
        """
        高级检索（阶段2完整功能）

        集成以下功能：
        1. Query Rewrite - LLM 优化查询
        2. Multi-Query Expansion - 多查询变体扩展
        3. 智能分类 - 基于模式的分类检索
        4. 阈值过滤 - 基于距离的质量过滤

        参数:
            query: str - 原始用户查询
            top_k: int - 返回结果数量
            enable_rewrite: bool - 是否启用查询重写（默认使用配置）
            enable_multi_query: bool - 是否启用多查询扩展（默认使用配置）
            enable_threshold: bool - 是否启用阈值过滤（默认使用配置）
            mode: str - 检索模式（默认使用配置）

        返回:
            Dict - 包含以下字段：
                - results: List[Dict] - 检索结果列表
                - original_query: str - 原始查询
                - rewritten_query: str | None - 重写后的查询
                - expanded_queries: List[str] | None - 扩展的查询列表
                - mode: str - 使用的检索模式
                - stats: Dict - 统计信息
        """
        if top_k is None:
            top_k = TOP_K_RESULTS
        if enable_rewrite is None:
            enable_rewrite = ENABLE_QUERY_REWRITE
        if enable_multi_query is None:
            enable_multi_query = ENABLE_MULTI_QUERY
        if enable_threshold is None:
            enable_threshold = ENABLE_THRESHOLD_FILTERING
        if mode is None:
            mode = RETRIEVAL_MODE

        # 统计信息
        stats = {
            "rewrite_enabled": enable_rewrite,
            "multi_query_enabled": enable_multi_query,
            "threshold_enabled": enable_threshold,
            "mode": mode
        }

        # Step 1: Query Rewrite
        rewritten_query = None
        working_query = query

        if enable_rewrite and self.llm is not None:
            rewritten_query = self._rewrite_query(query)
            working_query = rewritten_query
            stats["query_rewritten"] = (rewritten_query != query)
        else:
            stats["query_rewritten"] = False

        # Step 2: Multi-Query Expansion
        expanded_queries = None
        queries_to_search = [working_query]

        if enable_multi_query and self.llm is not None:
            expanded_queries = self._expand_queries(working_query)
            queries_to_search = expanded_queries
            stats["queries_expanded"] = len(expanded_queries)
        else:
            stats["queries_expanded"] = 1

        # Step 3: 确定分类
        category = None
        if mode == "keyword":
            category = self._classify_query_by_keywords(query)  # 使用原始查询分类
        elif mode == "metadata_only":
            category = self._extract_category_from_query(query)

        stats["category_filter"] = category

        # Step 4: 执行检索
        if len(queries_to_search) > 1:
            # 多查询检索
            results = self._retrieve_multi_query(
                queries_to_search,
                top_k=top_k,
                category_filter=category
            )
        else:
            # 单查询检索
            results = self.retrieve(
                queries_to_search[0],
                top_k=top_k * 2,  # 获取更多以便过滤
                category_filter=category
            )

        stats["raw_results_count"] = len(results)

        # Step 5: 阈值过滤
        if enable_threshold:
            threshold = RETRIEVAL_DISTANCE_THRESHOLD
            filtered = [r for r in results if r.get('distance', 999) < threshold]

            # 如果过滤后太少，放宽阈值
            if len(filtered) < 2 and len(results) > 0:
                relaxed_threshold = threshold * 1.5
                filtered = [r for r in results if r.get('distance', 999) < relaxed_threshold]
                stats["threshold_relaxed"] = True
            else:
                stats["threshold_relaxed"] = False

            results = filtered
            stats["filtered_results_count"] = len(results)

        # 限制返回数量
        final_results = results[:top_k]
        stats["final_results_count"] = len(final_results)

        return {
            "results": final_results,
            "original_query": query,
            "rewritten_query": rewritten_query,
            "expanded_queries": expanded_queries,
            "mode": mode,
            "stats": stats
        }

    def __repr__(self):
        doc_count = self.vectordb.count()
        return f"Retriever(documents={doc_count}, model={self.embedder.model_name})"


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("Retriever 模块测试")
    print("=" * 70)

    # 初始化
    retriever = Retriever()
    print(f"\n{retriever}\n")

    # 添加测试数据（如果数据库为空）
    if retriever.vectordb.count() == 0:
        print("数据库为空，添加测试数据...\n")

        from .ingestion import DocumentIngestion

        test_docs = [
            {
                "content": "示例政策文档：这是一个用于测试的政策类文档。包含规定和制度相关内容。",
                "metadata": {"category": "policies", "file": "sample_policy.md"}
            },
            {
                "content": "示例工作文档：这是关于工作流程和规范的文档。包含日常工作指南。",
                "metadata": {"category": "work", "file": "sample_work.md"}
            },
            {
                "content": "示例福利文档：这是关于员工福利的文档。包含各种福利政策说明。",
                "metadata": {"category": "benefits", "file": "sample_benefits.md"}
            }
        ]

        ingestion = DocumentIngestion(retriever.vectordb, retriever.embedder)
        for i, doc in enumerate(test_docs):
            ingestion.ingest_text(
                text=doc["content"],
                doc_id=f"test_doc_{i}",
                metadata=doc["metadata"]
            )

        print(f"✓ 已添加 {len(test_docs)} 条测试数据\n")

    # 测试检索
    queries = [
        "关于政策的问题？",
        "工作流程是什么？",
        "有什么福利？"
    ]

    for query in queries:
        print("=" * 70)
        print(f"查询: {query}")
        print("=" * 70)

        results = retriever.retrieve(query, top_k=2)

        for i, result in enumerate(results, 1):
            meta = result['metadata']
            doc = result['document']
            distance = result.get('distance', 'N/A')

            print(f"\n结果 {i}:")
            print(f"  来源: {meta['category']}/{meta['file']}")
            print(f"  相似度: {distance}")
            print(f"  内容: {doc[:80]}...")

        print()

    # 测试上下文组合
    print("\n" + "=" * 70)
    print("测试上下文组合")
    print("=" * 70)

    query = "关于政策的问题？"
    context = retriever.retrieve_with_context(query, top_k=2)

    print(f"\n查询: {query}\n")
    print("组合的上下文:")
    print(context)

    print("\n✓ 测试完成")
