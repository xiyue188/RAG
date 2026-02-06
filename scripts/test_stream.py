"""
测试流式输出功能
"""

import sys
import io
from pathlib import Path

# 彻底解决 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent))

from rag.llm import LLMClient
from rag.vectordb import VectorDB
from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.conversation import ConversationManager

def test_stream_output():
    """测试流式输出"""
    print("=" * 70)
    print("流式输出功能测试")
    print("=" * 70)

    # 初始化组件
    print("\n1. 初始化组件...")
    try:
        embedder = Embedder()
        vectordb = VectorDB()
        vectordb.get_collection()
        llm = LLMClient()
        retriever = Retriever(vectordb, embedder, llm)
        conversation = ConversationManager()
        print("✓ 所有组件初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return

    # 测试问题
    test_question = "什么是宠物政策？"
    print(f"\n2. 测试问题: {test_question}")

    # 检索
    print("\n3. 检索相关文档...")
    try:
        advanced_result = retriever.retrieve_advanced(test_question)
        results = advanced_result['results']
        print(f"[OK] 找到 {len(results)} 个文档")
    except Exception as e:
        print(f"[ERROR] 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试流式输出
    print("\n4. 测试流式输出...")
    print("-" * 70)

    try:
        # 获取对话上下文
        conversation_context = conversation.get_context_for_llm(max_turns=4)

        # 使用流式输出
        stream = llm.answer_smart_stream(
            test_question,
            results,
            conversation_context=conversation_context
        )

        # 第一个 yield 是元信息
        metadata = next(stream)
        print(f"元信息: mode={metadata['mode']}, reason={metadata['reason']}")
        print("\n回答:")
        print("-" * 70)

        # 流式输出答案
        full_answer = ""
        token_count = 0
        for token in stream:
            print(token, end='', flush=True)
            full_answer += token
            token_count += 1

        print("\n" + "-" * 70)
        print(f"✓ 流式输出成功！共 {token_count} 个token，总长度 {len(full_answer)} 字符")

    except Exception as e:
        print(f"\n✗ 流式输出失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    test_stream_output()
