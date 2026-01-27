"""
检索模块
负责语义检索逻辑
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .vectordb import VectorDB
from .embedder import Embedder
from config import TOP_K_RESULTS
from typing import List, Dict, Optional


class Retriever:
    """
    检索器
    封装语义检索逻辑
    """

    def __init__(self, vectordb: Optional[VectorDB] = None,
                 embedder: Optional[Embedder] = None):
        """
        初始化检索器

        参数:
            vectordb: VectorDB - 向量数据库实例（可选，默认创建新实例）
            embedder: Embedder - 向量化器实例（可选，默认创建新实例）
        """
        self.vectordb = vectordb or VectorDB()
        self.embedder = embedder or Embedder()

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
                "content": "TechCorp 宠物政策：员工可以在每周五带宠物来办公室。宠物必须性格温顺且已接种疫苗。",
                "metadata": {"category": "policies", "file": "pet_policy.md"}
            },
            {
                "content": "远程办公政策：员工每周最多可远程办公3天。核心工作时间为上午10点至下午3点。",
                "metadata": {"category": "policies", "file": "remote_work.md"}
            },
            {
                "content": "健康保险：公司提供全面的健康保险，包括医疗、牙科和视力保险。",
                "metadata": {"category": "benefits", "file": "health_insurance.md"}
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
        "可以带狗来公司吗？",
        "远程办公有什么规定？",
        "公司提供什么保险？"
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

    query = "宠物政策是什么？"
    context = retriever.retrieve_with_context(query, top_k=2)

    print(f"\n查询: {query}\n")
    print("组合的上下文:")
    print(context)

    print("\n✓ 测试完成")
