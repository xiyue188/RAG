"""
对比测试：严格模式 vs 混合模式
展示同一个问题在两种模式下的不同回答
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, LLMClient, VectorDB, Embedder
from config import SYSTEM_PROMPT_STRICT, SYSTEM_PROMPT_HYBRID
from dotenv import load_dotenv

print("=" * 70)
print("严格模式 vs 混合模式对比测试")
print("=" * 70)

load_dotenv()

# 初始化组件
vectordb = VectorDB()
embedder = Embedder()
retriever = Retriever(vectordb, embedder)

doc_count = vectordb.count()
print(f"\n数据库文档数: {doc_count}")

if doc_count == 0:
    print("\n✗ 数据库为空！请先运行: python scripts/2_ingest_docs.py")
    sys.exit(1)

# 测试问题（知识库内）
question = "可以带宠物来公司吗？"

print(f"\n问题: {question}\n")

# 检索
print("检索相关文档...")
results = retriever.retrieve(question, top_k=3)
context = retriever.retrieve_with_context(question, top_k=3)

print(f"✓ 找到 {len(results)} 个相关文档")
if results:
    print(f"  最高相似度: {results[0].get('distance', 'N/A'):.3f}")

print("\n" + "=" * 70)
print("【模式 1】严格模式（只用文档，不补充）")
print("=" * 70)

# 使用严格模式
llm_strict = LLMClient()
answer_strict = llm_strict.generate(
    prompt=f"""基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答：""",
    system_prompt=SYSTEM_PROMPT_STRICT
)

print("\n回答（严格模式）:")
print(answer_strict)

print("\n" + "=" * 70)
print("【模式 2】混合模式（文档优先 + 允许补充）")
print("=" * 70)

# 使用混合模式
llm_hybrid = LLMClient()
answer_hybrid = llm_hybrid.generate(
    prompt=f"""基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答：""",
    system_prompt=SYSTEM_PROMPT_HYBRID
)

print("\n回答（混合模式）:")
print(answer_hybrid)

print("\n" + "=" * 70)
print("对比分析")
print("=" * 70)
print("\n严格模式特点:")
print("  • 只基于文档内容")
print("  • 不添加任何背景知识")
print("  • 适合企业政策、法律文档等严肃场景")

print("\n混合模式特点:")
print("  • 优先使用文档内容")
print("  • 可以补充背景知识（会标注）")
print("  • 回答更丰富、更有帮助")
print("  • 适合客户服务、学习助手等场景")

print("\n" + "=" * 70)
print("✓ 对比测试完成")
print("=" * 70)
