"""可答性规则评测（M3 A1）

在 M1 的运行记录上比较多条判定规则，不重跑检索。方法见
`rag/eval/answerability.py` 模块说明。

用法：
  python scripts/eval_answerability.py
  python scripts/eval_answerability.py --out docs/eval/answerability_tables.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

for stream in (sys.stdout, sys.stderr):
    # 就地切 UTF-8，不替换流对象（见 STATUS 环境坑）
    stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag.answerability import BASIS_COSINE, BASIS_RERANK, AnswerabilityPolicy
from rag.eval.answerability import (
    RULES,
    UNANSWERABLE,
    Sample,
    bootstrap,
    build_samples,
    cross_validate,
    predict,
    rates,
    select_policy,
    undo_double_sigmoid,
)
from rag.eval.runner import read_results

RUNS = Path(__file__).parent.parent / "docs" / "eval" / "runs"
VARIANTS = [("rrf", BASIS_COSINE), ("vector_bge", BASIS_COSINE), ("rrf_rerank", BASIS_RERANK)]
VISIBILITY = [3, 20]
MAX_FRR = 0.10
BASELINE = "R0 top1"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(pair) -> str:
    return f"[{100 * pair[0]:.1f}, {100 * pair[1]:.1f}]"


def signed_ci(pair) -> str:
    return f"[{100 * pair[0]:+.1f}, {100 * pair[1]:+.1f}]"


def fmt_policy(p: AnswerabilityPolicy) -> str:
    parts = [f"{k.replace('min_', '')}≥{v:.3f}" for k, v in vars(p).items() if v is not None]
    return ", ".join(parts) or "—"


def stratified(samples: Sequence[Sample], preds: Sequence[bool]) -> Dict[str, tuple]:
    buckets = {"gold=1": lambda n: n == 1, "gold 2-4": lambda n: 2 <= n <= 4, "gold≥5": lambda n: n >= 5}
    out = {}
    for name, cond in buckets.items():
        idx = [i for i, s in enumerate(samples) if cond(s.n_gold)]
        sub = [samples[i] for i in idx]
        n_q = len({s.qid for s in sub})
        out[name] = (n_q, *rates(sub, [preds[i] for i in idx]))
    return out


def fold_range(policies: Sequence[AnswerabilityPolicy]) -> str:
    """各折选出的阈值范围。范围窄说明阈值稳定，宽说明对样本敏感。"""
    parts = []
    for field in vars(policies[0]):
        values = [getattr(p, field) for p in policies if getattr(p, field) is not None]
        if values:
            parts.append(f"{field.replace('min_', '')} {min(values):.3f}–{max(values):.3f}")
    return ", ".join(parts) or "—"


# (variant, k, rule) -> {(qid, label): 留出预测}，供跨方案配对比较
CV_PREDS: Dict[tuple, Dict[tuple, bool]] = {}


def evaluate(variant: str, basis: str, k: int, lines: List[str]) -> None:
    results = list(read_results(RUNS / f"{variant}.jsonl"))
    if basis == BASIS_RERANK:
        results = undo_double_sigmoid(results)  # M1 记录产自修复 4d933b8 之前
    samples = build_samples(results, k=k, basis=basis)
    n_ans = sum(1 for s in samples if s.label != UNANSWERABLE)
    n_unans = len(samples) - n_ans

    fixed = predict(samples, AnswerabilityPolicy(min_top1=config.ANSWERABLE_MIN_RELEVANCE))
    f_rec, f_frr = rates(samples, fixed)
    f_ci = bootstrap(samples, fixed)

    cv = {rule: cross_validate(samples, rule, max_frr=MAX_FRR) for rule in RULES}
    base = cv[BASELINE].preds

    lines += [
        f"### {variant} · 可见 top-{k}",
        "",
        f"样本：有答案 {n_ans}、无答案 {n_unans}。"
        f"生产固定阈值 `top1 ≥ {config.ANSWERABLE_MIN_RELEVANCE}`："
        f"识别率 {pct(f_rec)} {ci(f_ci['recall'])}，误拒率 {pct(f_frr)} {ci(f_ci['frr'])}。",
        "",
        "| 规则 | 识别率（CV） | 95% CI | 误拒率（CV） | 95% CI | 识别率 vs R0 | 配对差 95% CI | 样本内识别率 | 全量选参 | 各折选参范围 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule, res in cv.items():
        rec, frr = rates(samples, res.preds)
        b = bootstrap(samples, res.preds, baseline=None if rule == BASELINE else base)
        full = select_policy(samples, rule, max_frr=MAX_FRR)
        in_rec, _ = rates(samples, predict(samples, full))
        base_rec = rates(samples, base)[0]
        delta = "—" if rule == BASELINE else f"{100 * (rec - base_rec):+.1f}"
        delta_ci = "—" if rule == BASELINE else signed_ci(b["d_recall"])
        lines.append(
            f"| {rule} | {pct(rec)} | {ci(b['recall'])} | {pct(frr)} | {ci(b['frr'])} "
            f"| {delta} | {delta_ci} | {pct(in_rec)} | {fmt_policy(full)} | {fold_range(res.policies)} |"
        )
        CV_PREDS[(variant, k, rule)] = {(s.qid, s.label): p for s, p in zip(samples, res.preds)}

    best = max(cv, key=lambda r: rates(samples, cv[r].preds)[0])
    lines += ["", f"按 top-20 中 gold 数分层（{BASELINE} vs CV 识别率最高的 {best}）：", "",
              "| 分层 | query 数 | R0 识别率 | R0 误拒率 | 最优 识别率 | 最优 误拒率 |",
              "|---|---|---|---|---|---|"]
    s0, s1 = stratified(samples, base), stratified(samples, cv[best].preds)
    for name in s0:
        n_q, r0, e0 = s0[name]
        _, r1, e1 = s1[name]
        lines.append(f"| {name} | {n_q} | {pct(r0)} | {pct(e0)} | {pct(r1)} | {pct(e1)} |")
    lines.append("")

    print(f"[{variant} top-{k}] " + "  ".join(
        f"{r}={pct(rates(samples, c.preds)[0])}/{pct(rates(samples, c.preds)[1])}" for r, c in cv.items()
    ))


def cross_variant(a: tuple, b: tuple) -> List[str]:
    """两个方案的同一批 query 配对比较（按 qid+标签对齐，只取两边都有的样本）。"""
    pa, pb = CV_PREDS[a], CV_PREDS[b]
    keys = sorted(set(pa) & set(pb))
    samples = [Sample(q, label, None, 0) for q, label in keys]  # bootstrap 只用 qid 与标签
    preds_a, preds_b = [pa[key] for key in keys], [pb[key] for key in keys]
    ra, fa = rates(samples, preds_a)
    rb, fb = rates(samples, preds_b)
    d = bootstrap(samples, preds_a, baseline=preds_b)
    name = lambda t: f"{t[0]} top-{t[1]} {t[2]}"
    return [
        "### 跨方案配对比较",
        "",
        f"同一批 {len({q for q, _ in keys})} 条 query，按 qid 配对 bootstrap：",
        "",
        "| | 识别率 | 误拒率 |",
        "|---|---|---|",
        f"| {name(a)} | {pct(ra)} | {pct(fa)} |",
        f"| {name(b)} | {pct(rb)} | {pct(fb)} |",
        f"| 差（前 − 后） 95% CI | {signed_ci(d['d_recall'])} | {signed_ci(d['d_frr'])} |",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="可答性规则评测")
    parser.add_argument("--out", help="写入 Markdown")
    args = parser.parse_args()

    lines: List[str] = [
        "# 可答性规则评测 · 数据表",
        "",
        "> 由 `python scripts/eval_answerability.py --out docs/eval/answerability_tables.md` 生成，勿手改。",
        "> 分析与结论见 [answerability.md](answerability.md)。",
        "",
    ]
    for variant, basis in VARIANTS:
        for k in VISIBILITY:
            evaluate(variant, basis, k, lines)

    lines += cross_variant(("rrf", 20, "R2 gap_bg"), ("rrf_rerank", 20, BASELINE))

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[写入] {args.out}")
    else:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
