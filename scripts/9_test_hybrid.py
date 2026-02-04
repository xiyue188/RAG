"""
Hybrid 混合检索性能测试脚本
对比向量检索、BM25检索、Hybrid混合检索的效果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import Retriever, VectorDB, Embedder, LLMClient
from config import BM25_WEIGHT, VECTOR_WEIGHT


def print_results(title: str, results: list, max_display: int = 3):
    """打印检索结果"""
    print(f"\n{title}")
    print("-" * 70)

    if not results:
        print("  (无结果)")
        return

    for i, r in enumerate(results[:max_display], 1):
        # 提取分数信息
        score_info = []
        if 'hybrid_score' in r:
            score_info.append(f"Hybrid={r['hybrid_score']:.4f}")
        if 'vector_score' in r:
            score_info.append(f"Vector={r['vector_score']:.4f}")
        if 'bm25_score' in r:
            score_info.append(f"BM25={r['bm25_score']:.4f}")
        if 'distance' in r and 'hybrid_score' not in r:
            score_info.append(f"Distance={r['distance']:.4f}")
        if 'rerank_score' in r:
            score_info.append(f"Rerank={r['rerank_score']:.4f}")

        score_str = " | ".join(score_info)

        # 元数据
        category = r['metadata'].get('category', 'unknown')
        file = r['metadata'].get('file', 'unknown')

        # 内容预览
        content = r['document'][:80].replace('\n', ' ')

        print(f"  [{i}] {score_str}")
        print(f"      {category}/{file}")
        print(f"      {content}...")


def test_query(query: str, retriever: Retriever):
    """测试单个查询的不同检索方法"""
    print("\n" + "=" * 70)
    print(f"测试查询: '{query}'")
    print("=" * 70)

    # 1. 纯向量检索
    print("\n[1] 纯向量检索（Bi-Encoder）")
    vector_results = retriever.retrieve(query, top_k=3)
    print_results("结果:", vector_results)

    # 2. 纯 BM25 检索
    print("\n[2] 纯 BM25 关键词检索")
    bm25_results = retriever._bm25_search(query, top_k=3)
    print_results("结果:", bm25_results)

    # 3. Hybrid 混合检索（默认权重）
    print(f"\n[3] Hybrid 混合检索（权重: Vector={VECTOR_WEIGHT}, BM25={BM25_WEIGHT}）")
    hybrid_results = retriever._hybrid_search(query, top_k=3)
    print_results("结果:", hybrid_results)

    # 4. Hybrid + Rerank
    print("\n[4] Hybrid + Rerank 精排序")
    advanced_result = retriever.retrieve_advanced(
        query,
        top_k=3,
        enable_rewrite=False,
        enable_multi_query=False,
        enable_threshold=False,
        enable_rerank=True,
        enable_hybrid=True
    )
    print_results("结果:", advanced_result['results'])

    # 统计信息
    print("\n[统计信息]:")
    print(f"  向量检索结果数: {len(vector_results)}")
    print(f"  BM25 检索结果数: {len(bm25_results)}")
    print(f"  Hybrid 检索结果数: {len(hybrid_results)}")
    print(f"  Rerank 后结果数: {len(advanced_result['results'])}")

    return {
        'vector': vector_results,
        'bm25': bm25_results,
        'hybrid': hybrid_results,
        'hybrid_rerank': advanced_result['results']
    }


def test_weight_tuning(query: str, retriever: Retriever):
    """测试不同权重组合"""
    print("\n" + "=" * 70)
    print(f"权重调优测试: '{query}'")
    print("=" * 70)

    # 测试不同权重组合
    weight_configs = [
        (0.0, 1.0, "纯向量"),
        (0.3, 0.7, "向量为主"),
        (0.5, 0.5, "平衡"),
        (0.7, 0.3, "BM25为主"),
        (1.0, 0.0, "纯BM25")
    ]

    for bm25_w, vector_w, label in weight_configs:
        print(f"\n权重配置: {label} (BM25={bm25_w}, Vector={vector_w})")
        results = retriever._hybrid_search(
            query,
            top_k=3,
            bm25_weight=bm25_w,
            vector_weight=vector_w
        )

        if results:
            top1 = results[0]
            category = top1['metadata'].get('category', 'unknown')
            file = top1['metadata'].get('file', 'unknown')
            hybrid_score = top1.get('hybrid_score', 0)
            print(f"  Top-1: {category}/{file} (分数={hybrid_score:.4f})")
            print(f"    内容: {top1['document'][:60]}...")
        else:
            print("  (无结果)")


def main():
    print("=" * 70)
    print("Hybrid 混合检索性能测试")
    print("=" * 70)

    # 初始化组件
    print("\n初始化组件...")
    vectordb = VectorDB()
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(vectordb, embedder, llm)

    doc_count = vectordb.count()
    print(f"[OK] 向量数据库已加载（{doc_count} 个文档）")

    if doc_count == 0:
        print("\n[ERROR] 向量数据库为空，请先运行 2_ingest_data.py 导入文档")
        return

    # 测试查询集
    test_queries = [
        "401k 匹配比例是多少",         # 精确查询（适合向量）
        "401k match contribution",      # 英文关键词（适合BM25）
        "宠物政策的具体要求",          # 混合查询
        "远程办公申请流程",            # 流程类查询
    ]

    # 测试每个查询
    all_results = {}
    for query in test_queries:
        results = test_query(query, retriever)
        all_results[query] = results

    # 权重调优测试
    print("\n\n" + "=" * 70)
    print("权重调优实验")
    print("=" * 70)

    test_weight_tuning("401k 退休金匹配", retriever)

    # 总结
    print("\n\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    print("""
[检索方法对比]:

1. 纯向量检索（Bi-Encoder）
   优势：理解语义相似性，适合意图理解
   劣势：关键词精确匹配较弱

2. 纯 BM25 关键词检索
   优势：精确关键词匹配，适合术语查询
   劣势：不理解语义，同义词识别差

3. Hybrid 混合检索
   优势：综合两者优势，平衡语义和关键词
   劣势：需要权重调优

4. Hybrid + Rerank
   优势：最佳效果，Cross-Encoder精排
   劣势：延迟较高（+150ms）

[推荐配置]:
- 默认权重: Vector=0.7, BM25=0.3（语义为主）
- 精确查询: Vector=0.5, BM25=0.5（平衡）
- 关键词查询: Vector=0.3, BM25=0.7（关键词为主）

[推荐] 最佳实践：
   Hybrid + Rerank（适用于生产环境）
   """)

    print("\n[OK] 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
