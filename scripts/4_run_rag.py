"""
脚本4: 完整 RAG 流程
只调用 rag 模块，不包含逻辑
默认使用高级检索（阶段2）
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
    print("RAG 系统 - 完整流程（高级检索模式）")
    print("=" * 70)

    # 加载环境变量
    load_dotenv()

    # 1. 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()

    # 检查数据库
    doc_count = vectordb.count()
    print(f"数据库文档数: {doc_count}")

    if doc_count == 0:
        print("\n数据库为空！")
        print("  请先运行: python scripts/2_ingest_docs.py")
        return

    # 2. 初始化 LLM
    try:
        llm = LLMClient()
        print(f"LLM 初始化完成: {llm}")
    except Exception as e:
        print(f"\nLLM 初始化失败: {e}")
        print("\n请检查:")
        print("  1. .env 文件是否存在")
        print("  2. API Key 是否正确")
        print("  3. LLM_PROVIDER 设置是否正确")
        return

    # 3. 初始化检索器（传入 LLM 以支持高级检索）
    retriever = Retriever(vectordb, embedder, llm=llm)
    print("高级检索已启用（Query Rewrite + Multi-Query）")

    # 4. 交互式问答
    print("\n" + "=" * 70)
    print("RAG 问答系统已启动")
    print("=" * 70)
    print("\n命令:")
    print("  help    - 查看帮助")
    print("  status  - 查看状态")
    print("  quit    - 退出\n")

    while True:
        question = input("\n问题 > ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n再见！")
            break

        if question.lower() == 'help':
            print("\n可用命令:")
            print("  help       - 显示帮助")
            print("  status     - 查看数据库和检索状态")
            print("  categories - 查看文档分类")
            print("  quit       - 退出程序")
            continue

        if question.lower() == 'status':
            from config import RETRIEVAL_MODE
            print(f"\n状态:")
            print(f"  文档总数: {vectordb.count()}")
            print(f"  检索模式: {RETRIEVAL_MODE}")
            print(f"  高级检索: 已启用")
            print(f"    - Query Rewrite: 启用")
            print(f"    - Multi-Query: 启用")
            continue

        if question.lower() == 'categories':
            sample = vectordb.get(limit=100)
            if sample['metadatas']:
                cats = set(m.get('category', 'unknown') for m in sample['metadatas'])
                print(f"\n文档分类: {', '.join(sorted(cats))}")
            else:
                print("\n数据库中暂无文档")
            continue

        # RAG 流程 - 始终使用高级检索
        print("\n检索中...")

        advanced_result = retriever.retrieve_advanced(
            question,
            top_k=3,
            enable_rewrite=True,
            enable_multi_query=True
        )

        results = advanced_result['results']

        # 显示高级检索信息
        if advanced_result['rewritten_query'] and advanced_result['rewritten_query'] != question:
            print(f"  查询优化: '{question}'")
            print(f"         -> '{advanced_result['rewritten_query']}'")
        if advanced_result['expanded_queries']:
            print(f"  扩展查询: {len(advanced_result['expanded_queries'])} 个变体")

        # 显示检索结果
        if results:
            print(f"找到 {len(results)} 个相关文档:")
            for i, result in enumerate(results, 1):
                meta = result['metadata']
                distance = result.get('distance', 'N/A')
                if isinstance(distance, float):
                    print(f"  {i}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')} "
                          f"(距离: {distance:.3f})")
                else:
                    print(f"  {i}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}")
        else:
            print("未找到相关文档")

        # 生成答案
        print("\n生成答案...")

        try:
            result = llm.answer_smart(question, results)

            print("\n" + "=" * 70)
            print("回答:")
            print("=" * 70)
            print(result['answer'])
            print("=" * 70)
            print(f"\n[{result['mode']} | {result['reason']}]")

        except Exception as e:
            print(f"\n生成答案失败: {e}")


if __name__ == "__main__":
    main()
