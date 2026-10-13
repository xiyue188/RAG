"""
无答案阈值校准

**已被 `scripts/eval_answerability.py` 取代（M3 A1）。** 本脚本保留用于复现 M1
的原始报告，有两处已知偏差：0.05 步长会跨过拐点；有答案样本取
`relevances[0]` 而生产取最高分。新的结论与方法见 `docs/eval/answerability.md`。

解决 M0 遗留问题：ANSWERABLE_MIN_RELEVANCE 的初值 0.50 是拍定的，
且已被证明偏低（59 chunk 小语料上无答案查询得 0.692，被误判为可答）。

## 无答案子集的构造方式

取真实 query，把它的全部 gold 段落从检索结果中剔除，
剩下的 top1 relevance 就是"知识库没有这个知识点时"系统看到的分数。

为什么不手写一批"知识库肯定没有"的问题：写的时候就知道答案不在库里，
会不自觉写得过于离题，识别率虚高，线上不成立。而剔除 gold 的做法保留了
query 的真实性与领域相关性 —— 这才是线上的困难情形：
用户问了个合理问题，但知识库恰好没覆盖。

复用已有的 run 记录（docs/eval/runs/*.jsonl），不重跑检索。
每条记录里有完整的 retrieved_ids 与 relevances，剔除 gold 后
取剩余最高分即可，这是纯计算。

用法：
  python scripts/calibrate_threshold.py
  python scripts/calibrate_threshold.py --variant rrf --out docs/eval/threshold.md
"""

import argparse
import io
import sys
from pathlib import Path
from typing import List, Tuple

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag.eval.runner import read_results
from rag.eval.threshold import find_knee, format_ascii_curves, format_table, sweep

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "docs" / "eval" / "runs"

# 扫描范围。下界 0.30 以下已明显不可用（几乎不拒绝任何东西），
# 上界 0.90 以上误拒率必然过高。步长 0.05 足够看清拐点。
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90


def extract(variant: str) -> Tuple[List[float], List[float]]:
    """从 run 记录提取两组 top1 relevance。

    返回:
        (answerable_top1, unanswerable_top1)

    answerable：原样的 top1 relevance
    unanswerable：剔除全部 gold 后的 top1 relevance
                  —— 模拟"知识库没有这个知识点"
    """
    path = RUNS_DIR / f"{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少运行记录: {path}\n请先执行 python scripts/run_eval.py run"
        )

    answerable: List[float] = []
    unanswerable: List[float] = []

    for result in read_results(path):
        if result.error or not result.retrieved_ids:
            continue

        gold = set(result.gold_doc_ids)
        relevances = result.relevances

        # 有答案情形：直接取 top1
        answerable.append(relevances[0])

        # 无答案情形：剔除所有 gold，取剩余最高分。
        # 若剔除后为空（top-20 全是 gold），该条无法用于无答案评测，跳过 ——
        # 不能补 0.0，那会人为拉低分布使识别率虚高。
        remaining = [
            score
            for doc_id, score in zip(result.retrieved_ids, relevances)
            if doc_id not in gold
        ]
        if remaining:
            unanswerable.append(max(remaining))

    return answerable, unanswerable


def main() -> int:
    parser = argparse.ArgumentParser(description="无答案阈值校准")
    parser.add_argument("--variant", default="rrf", help="用哪个方案的记录校准")
    parser.add_argument(
        "--max-false-reject",
        type=float,
        default=0.10,
        help="可接受的误拒率上限（默认 0.10）",
    )
    parser.add_argument("--out", help="把结果写入 Markdown 文件")
    args = parser.parse_args()

    answerable, unanswerable = extract(args.variant)

    print("=" * 78)
    print(f"无答案阈值校准 · variant={args.variant}")
    print("=" * 78)
    print(f"  有答案样本: {len(answerable)} 条")
    print(f"  无答案样本: {len(unanswerable)} 条（剔除 gold 后仍有候选的条目）")
    print(f"  当前配置:   ANSWERABLE_MIN_RELEVANCE={config.ANSWERABLE_MIN_RELEVANCE}")

    if not answerable or not unanswerable:
        print("\n[!!] 样本不足，无法校准")
        return 1

    # 分布概览：两类的中位数差距决定阈值法是否可行
    def describe(values: List[float], label: str) -> str:
        ordered = sorted(values)
        n = len(ordered)
        return (
            f"  {label:<10} min={ordered[0]:.3f}  "
            f"p25={ordered[n // 4]:.3f}  "
            f"median={ordered[n // 2]:.3f}  "
            f"p75={ordered[3 * n // 4]:.3f}  "
            f"max={ordered[-1]:.3f}"
        )

    print("\nTop-1 relevance 分布:")
    print(describe(answerable, "有答案"))
    print(describe(unanswerable, "无答案"))

    overlap = sum(1 for v in unanswerable if v >= sorted(answerable)[len(answerable) // 2])
    print(
        f"\n  无答案样本中超过「有答案中位数」的比例: "
        f"{overlap / len(unanswerable):.1%}"
    )
    print("  这个比例越高，说明两类越难用单一阈值分开。")

    points = sweep(answerable, unanswerable, THRESHOLDS)

    print("\n" + format_ascii_curves(points))
    print()
    print(format_table(points))

    chosen = find_knee(points, max_false_reject=args.max_false_reject)
    print()
    print("=" * 78)
    print(f"建议阈值: {chosen.threshold:.2f}")
    print("=" * 78)
    print(f"  无答案识别率: {chosen.unanswerable_recall:.1%}")
    print(f"  误拒率:       {chosen.false_reject_rate:.1%}")
    print(f"  选点依据:     误拒率不超过 {args.max_false_reject:.0%} 的前提下识别率最高")
    print()
    print("  为什么用误拒率约束而非最大化平衡分：知识库问答场景中，")
    print("  误拒（用户明明能查到却被告知没有）的体验代价高于误答。")

    if abs(chosen.threshold - config.ANSWERABLE_MIN_RELEVANCE) > 0.01:
        print(
            f"\n  当前配置 {config.ANSWERABLE_MIN_RELEVANCE} 与建议值 "
            f"{chosen.threshold:.2f} 不一致，考虑更新 .env"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join([
            "# 无答案阈值校准",
            "",
            f"**方案**：{args.variant}",
            f"**有答案样本**：{len(answerable)} 条",
            f"**无答案样本**：{len(unanswerable)} 条",
            "",
            "## 构造方式",
            "",
            "取真实 query，把其全部 gold 段落从检索结果中剔除，",
            "剩余的 top1 relevance 即「知识库无此知识点」时系统看到的分数。",
            "不手写离题问题 —— 那会使识别率虚高且线上不成立。",
            "",
            "## 扫描结果",
            "",
            format_table(points),
            "",
            "## 建议",
            "",
            f"- 阈值：**{chosen.threshold:.2f}**",
            f"- 无答案识别率：{chosen.unanswerable_recall:.1%}",
            f"- 误拒率：{chosen.false_reject_rate:.1%}",
            f"- 依据：误拒率不超过 {args.max_false_reject:.0%} 的前提下识别率最高。",
            "  知识库问答场景中误拒的体验代价高于误答，故用约束而非最大化平衡分。",
            "",
        ])
        out_path.write_text(content, encoding="utf-8")
        print(f"\n[写入] {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
