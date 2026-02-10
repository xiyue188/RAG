"""
测试引用追踪功能
"""

import sys
import io
from pathlib import Path

# 解决编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent))

from rag.llm import LLMClient
from rag.vectordb import VectorDB
from rag.embedder import Embedder
from rag.retriever import Retriever


def test_citation_tracking():
    """测试引用追踪功能"""
    print("=" * 70)
    print("引用追踪功能测试")
    print("=" * 70)

    # 初始化组件
    print("\n1. 初始化组件...")
    try:
        embedder = Embedder()
        vectordb = VectorDB()
        vectordb.get_collection()
        llm = LLMClient()
        retriever = Retriever(vectordb, embedder, llm)
        print("[OK] 所有组件初始化成功")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        return

    # 测试问题
    test_question = "宠物政策有哪些具体规定？"
    print(f"\n2. 测试问题: {test_question}")

    # 检索相关文档
    print("\n3. 检索相关文档...")
    try:
        advanced_result = retriever.retrieve_advanced(test_question)
        results = advanced_result['results']
        print(f"[OK] 找到 {len(results)} 个文档")

        # 显示文档来源
        for i, result in enumerate(results[:3], 1):
            meta = result['metadata']
            print(f"  {i}. {meta.get('file', 'unknown')} - {result['document'][:50]}...")

    except Exception as e:
        print(f"[ERROR] 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试引用追踪
    print("\n4. 生成带引用的答案...")
    print("-" * 70)

    try:
        # 使用 answer_with_citations
        citation_result = llm.answer_with_citations(
            question=test_question,
            documents=results[:3],  # 只使用前3个最相关的文档
            format_style="inline"
        )

        # 显示结果
        print("\n原始答案:")
        print(citation_result['answer'])

        print("\n格式化后的答案（带引用）:")
        print(citation_result['formatted_answer'])

        print(f"\n引用统计:")
        print(f"  解析成功: {citation_result['parse_success']}")
        print(f"  引用数量: {len(citation_result.get('citations', []))}")

        # 显示详细引用信息
        if citation_result.get('citations'):
            print("\n详细引用:")
            for i, cite in enumerate(citation_result['citations'], 1):
                print(f"  {i}. 句子: {cite.sentence}")
                print(f"     来源: {cite.source_file} ({cite.source_doc})")

        print("\n" + "-" * 70)
        print("[OK] 引用追踪测试完成")

    except Exception as e:
        print(f"\n[ERROR] 引用生成失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_citation_tracking()
