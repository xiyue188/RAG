"""
脚本4: 完整 RAG 流程
只调用 rag 模块，不包含逻辑
默认使用高级检索（阶段2 + 阶段3 Rerank + Hybrid + Phase 1 对话管理）
"""

import sys
import io
from pathlib import Path

# 彻底解决 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, LLMClient, VectorDB, Embedder
from rag.conversation import ConversationManager, ReferenceResolver
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
    print("高级检索已启用（Multi-Query + Rerank + Hybrid）")

    # 4. 初始化对话管理器（Phase 1）
    conversation = ConversationManager(max_turns=20)
    resolver = ReferenceResolver(conversation)
    print("[OK] 对话管理已启用（支持多轮对话和指代消解）")

    # 5. 交互式问答
    print("\n" + "=" * 70)
    print("RAG 问答系统已启动")
    print("=" * 70)
    print("\n命令:")
    print("  help    - 查看帮助")
    print("  status  - 查看状态")
    print("  history - 查看对话历史")
    print("  clear   - 清空对话历史")
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
            print("  history    - 查看对话历史")
            print("  clear      - 清空对话历史")
            print("  quit       - 退出程序")
            continue

        if question.lower() == 'status':
            from config import (
                RETRIEVAL_MODE,
                ENABLE_MULTI_QUERY,
                ENABLE_RERANK,
                ENABLE_HYBRID,
                BM25_WEIGHT,
                VECTOR_WEIGHT,
                ENABLE_CITATION_TRACKING
            )
            print(f"\n状态:")
            print(f"  文档总数: {vectordb.count()}")
            print(f"  检索模式: {RETRIEVAL_MODE}")
            print(f"  对话历史: {len(conversation)} 轮")
            print(f"  高级检索: 已启用")
            print(f"    - Multi-Query: {'启用' if ENABLE_MULTI_QUERY else '禁用'}")
            print(f"    - Rerank精排: {'启用' if ENABLE_RERANK else '禁用'}")
            print(f"    - Hybrid混合: {'启用' if ENABLE_HYBRID else '禁用'}")
            if ENABLE_HYBRID:
                print(f"      权重: Vector={VECTOR_WEIGHT}, BM25={BM25_WEIGHT}")
            print(f"  引用追踪: {'启用' if ENABLE_CITATION_TRACKING else '禁用'}")
            continue

        if question.lower() == 'categories':
            sample = vectordb.get(limit=100)
            if sample['metadatas']:
                cats = set(m.get('category', 'unknown') for m in sample['metadatas'])
                print(f"\n文档分类: {', '.join(sorted(cats))}")
            else:
                print("\n数据库中暂无文档")
            continue

        if question.lower() == 'history':
            if len(conversation) == 0:
                print("\n暂无对话历史")
            else:
                print(f"\n对话历史（共 {len(conversation)} 轮）:")
                print("-" * 70)
                for i, turn in enumerate(conversation.get_recent_turns(10), 1):
                    role_label = "您" if turn.role == 'user' else "助手"
                    content = turn.content[:80] + "..." if len(turn.content) > 80 else turn.content
                    print(f"{i}. {role_label}: {content}")
            continue

        if question.lower() == 'clear':
            conversation.clear()
            print("\n[OK] 对话历史已清空")
            continue

        # 指代消解
        resolved_question = resolver.resolve(question)
        if resolved_question != question:
            print(f"  理解为: '{resolved_question}'")

        # 添加用户消息到历史（使用消解后的查询）
        conversation.add_user_message(resolved_question)

        # 获取对话上下文
        conversation_context = conversation.get_context_for_llm(max_turns=4)

        # RAG 流程 - 使用完整高级检索（阶段2 + 阶段3 + Phase 1）
        print("\n检索中...")

        advanced_result = retriever.retrieve_advanced(
            resolved_question,  # 使用消解后的查询
            top_k=5,  # 最终返回给LLM的文档数
            enable_multi_query=True,
            enable_rerank=True,
            enable_hybrid=True,
            conversation_context=conversation_context  # 传入对话上下文
        )

        results = advanced_result['results']
        stats = advanced_result['stats']

        # 显示高级检索信息
        # Query Rewrite 已删除（Phase 1 优化）
        if advanced_result['expanded_queries']:
            print(f"  扩展查询: {len(advanced_result['expanded_queries'])} 个变体")
        if stats.get('retrieval_method'):
            method_names = {
                'hybrid': 'Hybrid混合检索',
                'multi_query': '多查询检索',
                'vector_only': '向量检索'
            }
            print(f"  检索方法: {method_names.get(stats['retrieval_method'], stats['retrieval_method'])}")
        if stats.get('rerank_enabled') and stats.get('rerank_candidates', 0) > 0:
            print(f"  Rerank精排: {stats['rerank_candidates']} 个候选")

        # 显示检索结果
        if results:
            print(f"找到 {len(results)} 个相关文档:")
            for i, result in enumerate(results, 1):
                meta = result['metadata']
                # 显示分数信息
                score_info = []
                if 'hybrid_score' in result:
                    score_info.append(f"混合={result['hybrid_score']:.3f}")
                if 'rerank_score' in result:
                    score_info.append(f"精排={result['rerank_score']:.3f}")
                elif 'distance' in result:
                    score_info.append(f"距离={result['distance']:.3f}")

                score_str = f" ({', '.join(score_info)})" if score_info else ""
                print(f"  {i}. {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}{score_str}")
        else:
            print("未找到相关文档")

        # 生成答案
        print("\n生成答案...")

        try:
            from config import ENABLE_CITATION_TRACKING
            import re

            # 使用流式输出
            if ENABLE_CITATION_TRACKING and results:
                # 构建文档映射表（doc_1 -> 文件名）
                doc_map = {
                    f"doc_{i+1}": result['metadata'].get('file', 'unknown')
                    for i, result in enumerate(results)
                }

                # 启用引用追踪模式
                stream = llm.answer_with_citations_stream(
                    question,
                    results,
                    conversation_context=conversation_context
                )

                # 第一个 yield 是元信息
                metadata = next(stream)

                print("\n" + "=" * 70)
                print("回答（带来源标注）:")
                print("=" * 70)

                # 收集完整答案（用于保存到历史）
                full_answer = ""
                buffer = ""  # 用于累积token，检测完整的[doc_X]模式

                # 流式输出答案（实时替换内联引用标记）
                for item in stream:
                    if isinstance(item, dict) and item.get('type') == 'citation_meta':
                        # 引用元信息（最后返回）
                        # 先输出buffer中剩余内容
                        if buffer:
                            print(buffer, end='', flush=True)
                            full_answer += buffer
                            buffer = ""
                        citation_meta = item
                    else:
                        # 文本token - 累积到buffer中
                        buffer += item

                        # 尝试匹配并替换完整的[doc_X]
                        # 使用正则查找buffer中的[doc_X]
                        match = re.search(r'\[doc_(\d+)\]', buffer)
                        if match:
                            # 找到完整的[doc_X]，进行替换
                            doc_id = f"doc_{match.group(1)}"
                            file_name = doc_map.get(doc_id, 'unknown')

                            # 输出匹配前的内容 + 替换后的引用
                            pre_match = buffer[:match.start()]
                            replacement = f" [来源: {file_name}]"

                            print(pre_match + replacement, end='', flush=True)
                            full_answer += pre_match + replacement

                            # buffer保留匹配后的内容
                            buffer = buffer[match.end():]
                        elif len(buffer) > 20 and '[' not in buffer[-10:]:
                            # 如果buffer太长且最后10个字符没有'['，说明不会形成[doc_X]
                            # 输出前面的内容，保留最后10个字符
                            output = buffer[:-10]
                            print(output, end='', flush=True)
                            full_answer += output
                            buffer = buffer[-10:]

                print("\n" + "=" * 70)
                if 'citation_meta' in locals():
                    cited_count = citation_meta.get('cited_count', 0)
                    if cited_count > 0:
                        print(f"\n[引用追踪 | 引用了 {cited_count} 个文档]")
                    else:
                        print(f"\n[引用追踪 | 未找到引用标记]")
            else:
                # 标准模式（无引用追踪）
                stream = llm.answer_smart_stream(
                    question,
                    results,
                    conversation_context=conversation_context
                )

                # 第一个 yield 是元信息
                metadata = next(stream)

                print("\n" + "=" * 70)
                print("回答:")
                print("=" * 70)

                # 收集完整答案（用于保存到历史）
                full_answer = ""

                # 流式输出答案
                for token in stream:
                    print(token, end='', flush=True)
                    full_answer += token

                print("\n" + "=" * 70)
                print(f"\n[{metadata['mode']} | {metadata['reason']}]")

            # 添加助手回复到历史
            sources = [
                {
                    'file': r['metadata'].get('file'),
                    'category': r['metadata'].get('category')
                }
                for r in results
            ]
            conversation.add_assistant_message(full_answer, sources)

        except Exception as e:
            print(f"\n生成答案失败: {e}")


if __name__ == "__main__":
    main()
