"""
脚本4: 完整 RAG 流程
只调用 rag 模块，不包含逻辑
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, LLMClient, VectorDB, Embedder
from dotenv import load_dotenv


def main():
    """运行完整的 RAG 流程"""
    print("=" * 70)
    print("RAG 系统 - 完整流程")
    print("=" * 70)

    # 加载环境变量
    load_dotenv()

    # 1. 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()
    retriever = Retriever(vectordb, embedder)

    # 检查数据库
    doc_count = vectordb.count()
    print(f"✓ 数据库文档数: {doc_count}")

    if doc_count == 0:
        print("\n✗ 数据库为空！")
        print("  请先运行: python scripts/2_ingest_docs.py")
        return

    # 2. 初始化 LLM
    try:
        llm = LLMClient()
        print(f"✓ LLM 初始化完成: {llm}")
    except Exception as e:
        print(f"\n✗ LLM 初始化失败: {e}")
        print("\n请检查:")
        print("  1. .env 文件是否存在（复制 .env.example 并填写 API Key）")
        print("  2. API Key 是否正确")
        print("  3. LLM_PROVIDER 设置是否正确")
        return

    # 3. 交互式问答
    print("\n" + "=" * 70)
    print("RAG 问答系统已启动")
    print("=" * 70)
    print("\n提示:")
    print("  • 输入你的问题，系统会检索相关文档并生成答案")
    print("  • 输入 'quit' 或 'exit' 退出")
    print("  • 输入 'help' 查看帮助\n")

    while True:
        # 获取用户输入
        question = input("\n你的问题 > ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n再见！")
            break

        if question.lower() == 'help':
            print("\n可用命令:")
            print("  help  - 显示此帮助信息")
            print("  quit  - 退出程序")
            print("\n示例问题:")
            print("  • 可以带宠物来公司吗？")
            print("  • 远程办公有什么规定？")
            print("  • 公司有哪些福利？")
            continue

        # RAG 流程
        print("\n正在检索相关文档...")

        # Step 1: 检索
        results = retriever.retrieve(question, top_k=3)

        if not results:
            print("✗ 未找到相关文档")
            continue

        # 显示检索结果
        print(f"✓ 找到 {len(results)} 个相关文档:\n")
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            print(f"  {i}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}")

        # Step 2: 组合上下文
        context = retriever.retrieve_with_context(question, top_k=3)

        # Step 3: LLM 生成答案
        print("\n正在生成答案...")

        try:
            answer = llm.answer_with_context(question, context)

            print("\n" + "=" * 70)
            print("回答:")
            print("=" * 70)
            print(answer)
            print("=" * 70)

        except Exception as e:
            print(f"\n✗ 生成答案失败: {e}")


if __name__ == "__main__":
    main()
