"""
脚本7: 测试阶段2高级检索功能
测试 Query Rewrite 和 Multi-Query Expansion
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, LLMClient, VectorDB, Embedder
from dotenv import load_dotenv


def main():
    """测试阶段2高级检索功能"""
    print("=" * 70)
    print("阶段2: 高级检索功能测试")
    print("=" * 70)

    # 加载环境变量
    load_dotenv()

    # 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()

    doc_count = vectordb.count()
    print(f"数据库文档数: {doc_count}")

    if doc_count == 0:
        print("\n数据库为空！请先运行: python scripts/2_ingest_docs.py")
        return

    # 初始化 LLM
    try:
        llm = LLMClient()
        print(f"LLM 初始化完成: {llm}")
    except Exception as e:
        print(f"\nLLM 初始化失败: {e}")
        print("高级检索功能需要 LLM 支持")
        return

    # 初始化检索器（传入 LLM）
    retriever = Retriever(vectordb, embedder, llm=llm)
    print("Retriever 初始化完成（含 LLM 支持）")

    # 动态生成测试查询
    sample = vectordb.get(limit=3)
    test_queries = []

    if sample['documents']:
        # 从文档中提取关键词作为测试查询
        for doc in sample['documents'][:2]:
            # 取文档前20个字符作为查询基础
            query = doc[:30].strip()
            if query:
                test_queries.append(query)

    # 添加一些通用测试查询
    test_queries.extend([
        "这个系统是做什么的？",
        "有什么相关的政策？"
    ])

    # 测试1: Query Rewrite
    print("\n" + "=" * 70)
    print("测试1: Query Rewrite（查询重写）")
    print("=" * 70)

    for i, query in enumerate(test_queries[:2], 1):
        print(f"\n--- 测试 {i} ---")
        print(f"原始查询: {query}")

        rewritten = retriever._rewrite_query(query)
        print(f"重写查询: {rewritten}")

        if rewritten != query:
            print("结果: 查询已被重写")
        else:
            print("结果: 查询未变化（可能LLM未找到更好的表达）")

    # 测试2: Multi-Query Expansion
    print("\n" + "=" * 70)
    print("测试2: Multi-Query Expansion（多查询扩展）")
    print("=" * 70)

    for i, query in enumerate(test_queries[:2], 1):
        print(f"\n--- 测试 {i} ---")
        print(f"原始查询: {query}")

        variants = retriever._expand_queries(query, n=3)
        print(f"扩展查询 ({len(variants)} 个):")
        for j, v in enumerate(variants, 1):
            print(f"  {j}. {v}")

    # 测试3: 完整高级检索
    print("\n" + "=" * 70)
    print("测试3: retrieve_advanced（完整高级检索）")
    print("=" * 70)

    for i, query in enumerate(test_queries[:2], 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}: {query}")
        print("=" * 70)

        # 高级检索
        result = retriever.retrieve_advanced(
            query,
            top_k=3,
            enable_rewrite=True,
            enable_multi_query=True,
            enable_threshold=True
        )

        # 显示结果
        print(f"\n原始查询: {result['original_query']}")
        if result['rewritten_query']:
            print(f"重写查询: {result['rewritten_query']}")
        if result['expanded_queries']:
            print(f"扩展查询数: {len(result['expanded_queries'])}")

        print(f"\n检索统计:")
        stats = result['stats']
        for key, value in stats.items():
            print(f"  • {key}: {value}")

        print(f"\n检索结果 ({len(result['results'])} 个):")
        for j, r in enumerate(result['results'], 1):
            meta = r['metadata']
            distance = r.get('distance', 'N/A')
            if isinstance(distance, float):
                print(f"  {j}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')} "
                      f"(距离: {distance:.3f})")
            else:
                print(f"  {j}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}")

    # 测试4: 对比基础检索 vs 高级检索
    print("\n" + "=" * 70)
    print("测试4: 基础检索 vs 高级检索对比")
    print("=" * 70)

    comparison_query = test_queries[0] if test_queries else "相关政策"

    print(f"\n查询: {comparison_query}")

    # 基础检索
    print("\n--- 基础检索 (retrieve_smart) ---")
    basic_results = retriever.retrieve_smart(comparison_query, top_k=3)
    print(f"结果数: {len(basic_results)}")
    for j, r in enumerate(basic_results, 1):
        distance = r.get('distance', 'N/A')
        if isinstance(distance, float):
            print(f"  {j}. 距离: {distance:.3f}")
        else:
            print(f"  {j}. 距离: {distance}")

    # 高级检索
    print("\n--- 高级检索 (retrieve_advanced) ---")
    advanced_result = retriever.retrieve_advanced(
        comparison_query,
        top_k=3,
        enable_rewrite=True,
        enable_multi_query=True
    )
    print(f"结果数: {len(advanced_result['results'])}")
    for j, r in enumerate(advanced_result['results'], 1):
        distance = r.get('distance', 'N/A')
        if isinstance(distance, float):
            print(f"  {j}. 距离: {distance:.3f}")
        else:
            print(f"  {j}. 距离: {distance}")

    print("\n" + "=" * 70)
    print("阶段2高级检索测试完成")
    print("=" * 70)
    print("\n总结:")
    print("  • Query Rewrite: 使用 LLM 优化查询表达")
    print("  • Multi-Query: 生成多个查询变体提高召回率")
    print("  • retrieve_advanced: 集成所有高级功能")
    print("\n配置提示:")
    print("  • .env 中设置 ENABLE_QUERY_REWRITE=true 启用查询重写")
    print("  • .env 中设置 ENABLE_MULTI_QUERY=true 启用多查询扩展")
    print("  • .env 中设置 NUM_EXPANDED_QUERIES=3 调整扩展数量")


if __name__ == "__main__":
    main()
