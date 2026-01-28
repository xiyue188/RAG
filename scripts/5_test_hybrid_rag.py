"""
脚本5: 测试混合 RAG（知识库优先 + LLM 补充）
演示基于检索置信度的智能分流
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, LLMClient, VectorDB, Embedder
from config import SIMILARITY_THRESHOLD
from dotenv import load_dotenv


def main():
    """测试混合 RAG 功能"""
    print("=" * 70)
    print("混合 RAG 测试（知识库优先 + LLM 补充）")
    print("=" * 70)

    # 加载环境变量
    load_dotenv()

    # 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()
    retriever = Retriever(vectordb, embedder)
    llm = LLMClient()

    doc_count = vectordb.count()
    print(f"✓ 数据库文档数: {doc_count}")
    print(f"✓ LLM 初始化完成: {llm}")
    print(f"✓ 相似度阈值: {SIMILARITY_THRESHOLD}")

    if doc_count == 0:
        print("\n✗ 数据库为空！")
        print("  请先运行: python scripts/2_ingest_docs.py")
        return

    # 测试用例：包含在知识库内和知识库外的问题
    test_cases = [
        {
            "question": "可以带宠物来公司吗？",
            "expected_mode": "with_context",
            "description": "知识库内问题 - 应该使用文档回答"
        },
        {
            "question": "公司的远程办公政策是什么？",
            "expected_mode": "with_context",
            "description": "知识库内问题 - 应该使用文档回答"
        },
        {
            "question": "Python 中如何实现单例模式？",
            "expected_mode": "without_context",
            "description": "知识库外问题 - 应该使用 LLM 通用知识回答"
        },
        {
            "question": "什么是量子计算机？",
            "expected_mode": "without_context",
            "description": "知识库外问题 - 应该使用 LLM 通用知识回答"
        },
    ]

    print("\n" + "=" * 70)
    print(f"开始测试（共 {len(test_cases)} 个用例）")
    print("=" * 70)

    for i, test_case in enumerate(test_cases, 1):
        question = test_case['question']
        expected_mode = test_case['expected_mode']
        description = test_case['description']

        print(f"\n{'=' * 70}")
        print(f"测试 {i}/{len(test_cases)}: {description}")
        print(f"{'=' * 70}")
        print(f"问题: {question}\n")

        # Step 1: 检索
        print("🔍 检索相关文档...")
        results = retriever.retrieve(question, top_k=3)

        if results:
            print(f"✓ 找到 {len(results)} 个候选文档:")
            for j, result in enumerate(results, 1):
                meta = result['metadata']
                distance = result.get('distance', 'N/A')
                print(f"  {j}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')} "
                      f"(相似度距离: {distance:.3f})")
        else:
            print("✗ 未找到相关文档")

        # Step 2: 智能回答（自动分流）
        print("\n💡 生成答案...")
        try:
            result = llm.answer_smart(question, results)

            # 显示结果
            print(f"\n{'=' * 70}")
            print("回答结果:")
            print(f"{'=' * 70}")
            print(f"模式: {result['mode']}")
            print(f"原因: {result['reason']}")
            if result['max_similarity'] is not None:
                print(f"最高相似度: {result['max_similarity']:.3f}")
            print(f"相关文档数: {result['relevant_docs_count']}")
            print(f"\n回答:\n{result['answer']}")
            print(f"{'=' * 70}")

            # 验证预期
            if result['mode'] == expected_mode:
                print(f"✅ 通过：模式符合预期（{expected_mode}）")
            else:
                print(f"⚠️  警告：预期模式 {expected_mode}，实际模式 {result['mode']}")

        except Exception as e:
            print(f"\n✗ 生成答案失败: {e}")

    print("\n" + "=" * 70)
    print("✓ 混合 RAG 测试完成")
    print("=" * 70)
    print("\n关键要点:")
    print(f"  • 相似度阈值: {SIMILARITY_THRESHOLD}")
    print(f"  • 距离 < {SIMILARITY_THRESHOLD} → 使用文档回答（with_context）")
    print(f"  • 距离 >= {SIMILARITY_THRESHOLD} → 使用 LLM 通用知识（without_context）")
    print("  • LLM 会标注哪些内容来自通用知识")


if __name__ == "__main__":
    main()
