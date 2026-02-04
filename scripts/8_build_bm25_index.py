"""
BM25 索引构建脚本
显式构建 BM25 索引并验证
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, VectorDB, Embedder, LLMClient


def main():
    print("=" * 70)
    print("BM25 索引构建")
    print("=" * 70)

    # 初始化组件
    print("\n[1/4] 初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(vectordb, embedder, llm)

    doc_count = vectordb.count()
    print(f"[OK] 向量数据库已加载（{doc_count} 个文档）")

    if doc_count == 0:
        print("\n[ERROR] 向量数据库为空，请先运行 2_ingest_data.py 导入文档")
        return

    # 构建 BM25 索引
    print("\n[2/4] 构建 BM25 索引...")
    bm25_index = retriever._build_bm25_index()

    if bm25_index is None:
        print("[ERROR] BM25 索引构建失败")
        return

    print(f"[OK] BM25 索引构建成功")
    print(f"  - 索引文档数: {len(retriever._bm25_docs)}")
    print(f"  - 元数据数量: {len(retriever._bm25_metadatas)}")

    # 验证索引
    print("\n[3/4] 验证 BM25 索引...")

    # 测试查询
    test_queries = [
        "401k 退休金",
        "宠物政策",
        "远程办公"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n测试查询 {i}: '{query}'")
        results = retriever._bm25_search(query, top_k=3)

        if results:
            print(f"  [OK] 找到 {len(results)} 个结果")
            for j, r in enumerate(results, 1):
                score = r.get('bm25_score', 0)
                category = r['metadata'].get('category', 'unknown')
                file = r['metadata'].get('file', 'unknown')
                doc_preview = r['document'][:50] + "..."
                print(f"    [{j}] 分数={score:.4f} | {category}/{file}")
                print(f"        内容: {doc_preview}")
        else:
            print("  [FAIL] 没有找到结果")

    # 统计信息
    print("\n[4/4] BM25 索引统计信息")
    print("=" * 70)
    print(f"总文档数: {len(retriever._bm25_docs)}")
    print(f"索引大小: 约 {sys.getsizeof(bm25_index) / 1024:.2f} KB")
    print("\n[OK] BM25 索引构建和验证完成")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
