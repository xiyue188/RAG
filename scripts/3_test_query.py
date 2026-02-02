"""
脚本3: 测试检索
只调用 rag 模块，不包含逻辑
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, VectorDB, Embedder


def main():
    """测试检索功能"""
    print("=" * 70)
    print("语义检索测试")
    print("=" * 70)

    # 初始化组件
    vectordb = VectorDB()
    embedder = Embedder()
    retriever = Retriever(vectordb, embedder)

    # 检查数据库
    doc_count = vectordb.count()
    print(f"\n数据库文档数: {doc_count}")

    if doc_count == 0:
        print("\n✗ 数据库为空！")
        print("  请先运行: python scripts/2_ingest_docs.py")
        return

    # 动态生成测试查询
    def get_sample_queries(vdb, n=3):
        """从数据库动态生成测试查询"""
        sample = vdb.get(limit=10)
        queries = []
        for doc in sample['documents'][:n]:
            # 取文档前20字作为查询
            query = doc[:20].strip() + "?"
            queries.append(query)
        return queries

    test_queries = get_sample_queries(vectordb, n=3)
    print(f"\n动态生成 {len(test_queries)} 个测试查询")
    print("（从数据库文档中提取前20字作为查询）\n")

    for i, query in enumerate(test_queries, 1):
        print("=" * 70)
        print(f"查询 {i}: {query}")
        print("=" * 70)

        # 检索
        results = retriever.retrieve(query, top_k=3)

        # 显示结果
        for j, result in enumerate(results, 1):
            meta = result['metadata']
            doc = result['document']
            distance = result.get('distance', 'N/A')

            print(f"\n结果 {j}:")
            print(f"  来源: {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}")
            print(f"  相似度距离: {distance}")
            print(f"  内容: {doc[:100]}...")

        print()

    print("=" * 70)
    print("✓ 检索测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
