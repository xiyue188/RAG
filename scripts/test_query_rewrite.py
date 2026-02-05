"""
测试改进后的查询重写功能
验证保守重写策略是否解决了检索退化问题
"""

import sys
import io
from pathlib import Path

# 设置标准输出编码为UTF-8（永久修复Windows GBK编码问题）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, VectorDB, Embedder, LLMClient


def test_query_rewrite_improvement():
    """测试改进后的查询重写"""
    print("=" * 70)
    print("查询重写改进测试")
    print("=" * 70)

    # 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(vectordb, embedder, llm)

    doc_count = vectordb.count()
    print(f"向量数据库已加载（{doc_count} 个文档）")

    if doc_count == 0:
        print("\n[ERROR] 向量数据库为空，请先运行 2_ingest_data.py 导入文档")
        return

    # 测试用例
    test_queries = [
        "什么是宠物政策？",
        "401k匹配比例是多少？",
        "远程办公的申请流程",
    ]

    print("\n" + "=" * 70)
    print("测试：查询重写对检索质量的影响")
    print("=" * 70)

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"原始查询: '{query}'")
        print("=" * 70)

        # 1. 不重写的检索结果
        print("\n[1] 不重写 - 直接检索:")
        results_no_rewrite = retriever.retrieve(query, top_k=3)

        for i, r in enumerate(results_no_rewrite, 1):
            category = r['metadata'].get('category', 'unknown')
            file = r['metadata'].get('file', 'unknown')
            distance = r.get('distance', 'N/A')
            print(f"  #{i} {category}/{file} (距离: {distance:.4f})")

        # 2. 使用改进的重写
        print("\n[2] 改进的查询重写 (保守策略):")
        rewritten = retriever._rewrite_query(query)
        print(f"  重写后: '{rewritten}'")

        results_rewrite = retriever.retrieve(rewritten, top_k=3)

        for i, r in enumerate(results_rewrite, 1):
            category = r['metadata'].get('category', 'unknown')
            file = r['metadata'].get('file', 'unknown')
            distance = r.get('distance', 'N/A')
            print(f"  #{i} {category}/{file} (距离: {distance:.4f})")

        # 3. 对比分析
        print("\n[3] 对比分析:")

        # 检查Top-1是否一致
        if results_no_rewrite and results_rewrite:
            top1_no_rewrite = results_no_rewrite[0]['metadata'].get('file')
            top1_rewrite = results_rewrite[0]['metadata'].get('file')

            dist_no_rewrite = results_no_rewrite[0].get('distance', float('inf'))
            dist_rewrite = results_rewrite[0].get('distance', float('inf'))

            if top1_no_rewrite == top1_rewrite:
                print(f"  ✓ Top-1 文档一致: {top1_no_rewrite}")
                if dist_rewrite <= dist_no_rewrite * 1.1:  # 允许10%的误差
                    print(f"  ✓ 距离保持稳定: {dist_no_rewrite:.4f} → {dist_rewrite:.4f}")
                else:
                    print(f"  ⚠️ 距离变差: {dist_no_rewrite:.4f} → {dist_rewrite:.4f}")
            else:
                print(f"  ✗ Top-1 文档改变: {top1_no_rewrite} → {top1_rewrite}")
                print(f"  ✗ 距离变化: {dist_no_rewrite:.4f} → {dist_rewrite:.4f}")
                print(f"  → 查询重写可能导致检索退化")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
改进的查询重写策略（保守原则）：
1. ✓ 保留所有核心关键词
2. ✓ 仅去除口语化表达（"什么是"、"请问"等）
3. ✓ 不添加额外信息（公司名称、时间等）
4. ✓ 保持查询简洁

预期效果：
- Top-1 文档排序保持稳定
- 距离分数不应显著变差
- 避免过度优化导致的检索退化
    """)


def test_context_aware_rewrite():
    """测试上下文感知的查询重写"""
    print("\n" + "=" * 70)
    print("上下文感知重写测试")
    print("=" * 70)

    from rag.conversation import ConversationManager

    # 初始化
    vectordb = VectorDB()
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(vectordb, embedder, llm)

    conv = ConversationManager()
    conv.add_user_message("什么是宠物政策？")
    conv.add_assistant_message("TechCorp允许员工携带宠物上班...")

    # 测试指代消解结合重写
    followup_query = "它还有什么要求？"
    context = conv.get_context_for_llm(max_turns=4)

    print(f"\n对话上下文:")
    print("-" * 70)
    print(context)
    print("-" * 70)

    print(f"\n后续查询: '{followup_query}'")

    # 使用上下文重写
    rewritten = retriever._rewrite_query(followup_query, conversation_context=context)
    print(f"重写后: '{rewritten}'")

    # 检索
    results = retriever.retrieve(rewritten, top_k=3)
    print(f"\n检索结果:")
    for i, r in enumerate(results, 1):
        category = r['metadata'].get('category', 'unknown')
        file = r['metadata'].get('file', 'unknown')
        distance = r.get('distance', 'N/A')
        print(f"  #{i} {category}/{file} (距离: {distance:.4f})")

    # 验证是否找到宠物政策相关文档
    found_pet_policy = any('pet' in r['metadata'].get('file', '').lower()
                           for r in results)

    if found_pet_policy:
        print("\n  ✓ 成功找到宠物政策相关文档")
    else:
        print("\n  ✗ 未找到宠物政策相关文档（可能需要进一步优化）")


def main():
    """运行所有测试"""
    try:
        test_query_rewrite_improvement()
        test_context_aware_rewrite()

        print("\n" + "=" * 70)
        print("✓ 所有测试完成")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
