"""
脚本7: 测试阶段3 Rerank 功能
对比有无 Rerank 的检索效果
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, LLMClient, VectorDB, Embedder
from dotenv import load_dotenv


def main():
    print("=" * 70)
    print("阶段3: Rerank 精排序测试")
    print("=" * 70)

    # 加载环境变量
    load_dotenv()

    # 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(vectordb, embedder, llm=llm)

    # 检查数据库
    doc_count = vectordb.count()
    print(f"数据库文档数: {doc_count}")

    if doc_count == 0:
        print("\n数据库为空！")
        print("  请先运行: python scripts/2_ingest_docs.py")
        return

    # 测试查询列表
    test_queries = [
        "相关政策规定",
        "员工福利",
        "工作流程"
    ]

    print("\n" + "=" * 70)
    print("开始对比测试")
    print("=" * 70)

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"查询: {query}")
        print('=' * 70)

        # 测试1: 无 Rerank
        print("\n【测试1: 无 Rerank】")
        result_no_rerank = retriever.retrieve_advanced(
            query,
            top_k=5,
            enable_rerank=False
        )

        print(f"结果数: {len(result_no_rerank['results'])}")
        if result_no_rerank['results']:
            print("Top 3 结果:")
            for i, r in enumerate(result_no_rerank['results'][:3], 1):
                distance = r.get('distance', 'N/A')
                if isinstance(distance, float):
                    print(f"  {i}. ID: {r['id'][:20]}... | 距离: {distance:.3f}")
                else:
                    print(f"  {i}. ID: {r['id'][:20]}...")
        else:
            print("  未找到结果")

        # 测试2: 启用 Rerank
        print("\n【测试2: 启用 Rerank】")
        result_rerank = retriever.retrieve_advanced(
            query,
            top_k=5,
            enable_rerank=True
        )

        print(f"结果数: {len(result_rerank['results'])}")
        print(f"统计信息: {result_rerank['stats']}")
        if result_rerank['results']:
            print("Top 3 结果:")
            for i, r in enumerate(result_rerank['results'][:3], 1):
                rerank_score = r.get('rerank_score', 'N/A')
                if isinstance(rerank_score, float):
                    print(f"  {i}. ID: {r['id'][:20]}... | Rerank分数: {rerank_score:.3f}")
                else:
                    print(f"  {i}. ID: {r['id'][:20]}...")
        else:
            print("  未找到结果")

        # 对比结果
        if result_no_rerank['results'] and result_rerank['results']:
            print("\n【排序对比】")
            print(f"  无Rerank Top1: {result_no_rerank['results'][0]['id'][:30]}...")
            print(f"  有Rerank Top1: {result_rerank['results'][0]['id'][:30]}...")

            # 检查是否有排序变化
            no_rerank_ids = [r['id'] for r in result_no_rerank['results'][:3]]
            rerank_ids = [r['id'] for r in result_rerank['results'][:3]]

            if no_rerank_ids != rerank_ids:
                print("  ✓ Rerank 改变了排序")
            else:
                print("  - Rerank 未改变排序（结果一致）")

        print()

    # 详细展示一个查询的完整结果
    print("\n" + "=" * 70)
    print("详细结果展示（单个查询）")
    print("=" * 70)

    detailed_query = test_queries[0]
    print(f"\n查询: {detailed_query}\n")

    # 无 Rerank
    print("【无 Rerank - 前3个结果】")
    result = retriever.retrieve_advanced(
        detailed_query,
        top_k=5,
        enable_rerank=False
    )

    for i, r in enumerate(result['results'][:3], 1):
        meta = r['metadata']
        doc = r['document']
        distance = r.get('distance', 'N/A')

        print(f"\n结果 {i}:")
        print(f"  来源: {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}")
        if isinstance(distance, float):
            print(f"  距离: {distance:.4f}")
        print(f"  内容: {doc[:100]}...")

    # 启用 Rerank
    print("\n" + "-" * 70)
    print("【启用 Rerank - 前3个结果】")
    result = retriever.retrieve_advanced(
        detailed_query,
        top_k=5,
        enable_rerank=True
    )

    for i, r in enumerate(result['results'][:3], 1):
        meta = r['metadata']
        doc = r['document']
        rerank_score = r.get('rerank_score', 'N/A')

        print(f"\n结果 {i}:")
        print(f"  来源: {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}")
        if isinstance(rerank_score, float):
            print(f"  Rerank分数: {rerank_score:.4f}")
        print(f"  内容: {doc[:100]}...")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print("\n总结:")
    print("  ✓ Rerank 使用 Cross-Encoder 对初步检索结果进行精排序")
    print("  ✓ 可以提高最相关文档的排序位置")
    print("  ✓ 适用于需要高精度排序的场景")
    print()


if __name__ == "__main__":
    main()
