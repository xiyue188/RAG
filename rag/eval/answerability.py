"""可答性评测：在 M1 的运行记录上比较多条判定规则

## 样本构造（沿用 M1 的无答案构造法）

每条 query 产出一对样本：

- **有答案**：原始检索结果的前 k 条
- **无答案**：把该 query 的全部 gold 从结果中剔除后的前 k 条
  —— 模拟"用户问了合理问题，但知识库恰好没覆盖这个知识点"

k 取两种可见性：

- **3**：生产当前交给判断函数的条数（`TOP_K_RESULTS=3`）
- **20**：检索内部的候选池（`HYBRID_TOP_K=20`）

前者回答"不改接口能做到多少"，后者回答"把候选分数传下去值不值"。

## 0.0 视为缺失（仅余弦口径）

run 记录只存了 relevance 浮点数，丢了 `has_relevance_signal` 标记。
RRF 方案里只被 BM25 召回的条目 relevance 为 0.0（无向量距离）。
余弦口径按 `0.0 ⇒ 缺失` 还原：真实的 0.0 需要两向量方向相反，实际不出现。
rerank 口径下每条候选都被精排过、没有缺失，且概率可以合法地接近 0，不做此还原。

## M1 的 rerank 记录要先还原

修复 `4d933b8` 之前，rerank 分数被 sigmoid 了两次，记录里存的是
`s = sigmoid(p)`（p 才是模型给出的相关概率），值域 [0.5, 0.731]。
`undo_double_sigmoid` 精确还原 `p = ln(s / (1 - s))`。不还原也能得到同样
的识别率（单调变换不改变排序），但阈值会落在错误的尺度上，无法用于生产。

## 选参与计分分离

- 按 qid 分组的 k 折交叉验证：在训练折上选参，在留出折上计分。
  **同一 query 的两个样本必须在同一折** —— 否则参数在测试样本的
  配对样本上被调过，等于泄漏
- 选参目标：误拒率 ≤ 上限的前提下识别率最高（与 M1 同一纪律）
- 置信区间：按 qid 的 bootstrap；与基线的差用**配对** bootstrap

## 速度与一致性

二维网格 × 5 折 × 多方案，逐个调用 `decide` 太慢，选参用 numpy 向量化。
**留出折上的最终预测调用生产的 `decide`** —— 评测量的必须是生产的行为。
两者一致性由 `tests/test_answerability_eval.py` 守着。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..answerability import (
    BASIS_COSINE,
    BASIS_NONE,
    AnswerabilityPolicy,
    AnswerabilitySignals,
    decide,
    extract_signals,
)
from .runner import QueryResult

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class Sample:
    qid: str
    label: str  # ANSWERABLE / UNANSWERABLE
    signals: AnswerabilitySignals
    # 原始 top-20 中的 gold 数，用于分层报告
    n_gold: int


# 规则族：名字 → 使用哪些策略字段
RULES: Dict[str, Tuple[str, ...]] = {
    "R0 top1": ("min_top1",),
    "R1 top1+gap12": ("min_top1", "min_gap12"),
    "R2 gap_bg": ("min_gap_bg",),
    "R3 top1+gap_bg": ("min_top1", "min_gap_bg"),
}
_FEATURE_OF = {"min_top1": "top1", "min_gap12": "gap12", "min_gap_bg": "gap_bg"}


def undo_double_sigmoid(results: Iterable[QueryResult]) -> List[QueryResult]:
    """还原修复前 rerank 记录的分数：p = ln(s / (1 - s))。见模块说明。"""
    fixed = []
    for r in results:
        rel = [math.log(s / (1.0 - s)) if 0.0 < s < 1.0 else s for s in r.relevances]
        fixed.append(replace(r, relevances=[round(p, 6) for p in rel]))
    return fixed


def build_samples(
    results: Iterable[QueryResult], k: int, basis: str = BASIS_COSINE
) -> List[Sample]:
    """从 run 记录构造配对样本。

    剔除 gold 后结果为空的 query 不产出无答案样本：不能用"空列表"代表
    "知识库没有" —— 那会人为抬高识别率（与 M1 同一处理）。
    """
    samples: List[Sample] = []
    for r in results:
        if r.error or not r.retrieved_ids:
            continue
        gold = set(r.gold_doc_ids)
        ranked = list(zip(r.retrieved_ids, r.relevances))
        n_gold = sum(1 for doc_id, _ in ranked if doc_id in gold)

        samples.append(Sample(r.qid, ANSWERABLE, _signals(ranked[:k], basis), n_gold))

        residual = [(d, s) for d, s in ranked if d not in gold]
        if residual:
            samples.append(
                Sample(r.qid, UNANSWERABLE, _signals(residual[:k], basis), n_gold)
            )
    return samples


def _signals(ranked: Sequence[Tuple[str, float]], basis: str) -> AnswerabilitySignals:
    if basis == BASIS_COSINE:
        scores = [s for _, s in ranked if s > 0.0]  # 0.0 ⇒ 缺失，见模块说明
    else:
        scores = [s for _, s in ranked]
    # 前 k 条全部无信号：生产里 assess 会走"无信号 → 信任排序"分支
    return extract_signals(scores, basis if scores else BASIS_NONE)


# ============================================================
# 预测与计分
# ============================================================

def predict(samples: Sequence[Sample], policy: AnswerabilityPolicy) -> List[bool]:
    """用生产的 decide 逐条判定。True = 判为可答。"""
    return [decide(s.signals, policy).answerable for s in samples]


def rates(samples: Sequence[Sample], preds: Sequence[bool]) -> Tuple[float, float]:
    """(无答案识别率, 误拒率)。"""
    unans = [p for s, p in zip(samples, preds) if s.label == UNANSWERABLE]
    ans = [p for s, p in zip(samples, preds) if s.label == ANSWERABLE]
    recall = sum(1 for p in unans if not p) / len(unans) if unans else 0.0
    frr = sum(1 for p in ans if not p) / len(ans) if ans else 0.0
    return recall, frr


class _Matrix:
    """特征的 numpy 视图，供网格搜索向量化。

    语义与 `decide` 对齐：无信号 → 可答；无结果 → 不可答；
    特征为 None（如只有一条结果时的分差）→ 该条件不据此拒答。
    """

    def __init__(self, samples: Sequence[Sample]):
        self.no_signal = np.array([s.signals.basis == BASIS_NONE for s in samples])
        self.no_result = np.array(
            [s.signals.basis != BASIS_NONE and s.signals.top1 is None for s in samples]
        )
        self.is_unans = np.array([s.label == UNANSWERABLE for s in samples])
        self.features = {
            name: np.array(
                [np.nan if getattr(s.signals, name) is None else getattr(s.signals, name)
                 for s in samples],
                dtype=float,
            )
            for name in ("top1", "gap12", "gap_bg")
        }

    def predict(self, policy: AnswerabilityPolicy) -> np.ndarray:
        ok = np.ones(len(self.is_unans), dtype=bool)
        for field, feature in _FEATURE_OF.items():
            threshold = getattr(policy, field)
            if threshold is None:
                continue
            values = self.features[feature]
            # NaN（特征不可评估）不据此拒答
            ok &= np.isnan(values) | (values >= threshold)
        ok &= ~self.no_result
        ok |= self.no_signal
        return ok


def _candidates(values: np.ndarray, n: int) -> List[float]:
    """候选阈值取特征的分位点：与尺度无关，余弦与 rerank 口径共用。"""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return []
    return sorted({float(q) for q in np.quantile(finite, np.linspace(0, 1, n))})


def select_policy(
    samples: Sequence[Sample],
    rule: str,
    max_frr: float = 0.10,
    grid: Optional[int] = None,
) -> AnswerabilityPolicy:
    """在给定样本上为规则选参：误拒率 ≤ max_frr 时识别率最高。

    同识别率取误拒率低者；仍并列取阈值小者（更保守，少拒答）。
    """
    fields = RULES[rule]
    grid = grid or (101 if len(fields) == 1 else 41)
    m = _Matrix(samples)
    axes = [_candidates(m.features[_FEATURE_OF[f]], grid) for f in fields]

    best_key, best_policy = None, AnswerabilityPolicy()
    for combo in np.array(np.meshgrid(*axes)).T.reshape(-1, len(fields)):
        policy = AnswerabilityPolicy(**dict(zip(fields, (float(v) for v in combo))))
        ok = m.predict(policy)
        recall = float((~ok & m.is_unans).sum() / max(m.is_unans.sum(), 1))
        frr = float((~ok & ~m.is_unans).sum() / max((~m.is_unans).sum(), 1))
        if frr > max_frr:
            continue
        key = (recall, -frr, *(-float(v) for v in combo))
        if best_key is None or key > best_key:
            best_key, best_policy = key, policy
    return best_policy


# ============================================================
# 交叉验证与置信区间
# ============================================================

@dataclass
class CVResult:
    rule: str
    preds: List[bool]                    # 与 samples 一一对应的留出预测
    policies: List[AnswerabilityPolicy]  # 每折选出的参数


def cross_validate(
    samples: Sequence[Sample],
    rule: str,
    folds: int = 5,
    seed: int = 0,
    max_frr: float = 0.10,
) -> CVResult:
    qids = sorted({s.qid for s in samples})
    random.Random(seed).shuffle(qids)
    fold_of = {qid: i % folds for i, qid in enumerate(qids)}

    preds: List[Optional[bool]] = [None] * len(samples)
    policies: List[AnswerabilityPolicy] = []
    for f in range(folds):
        train = [s for s in samples if fold_of[s.qid] != f]
        policy = select_policy(train, rule, max_frr=max_frr)
        policies.append(policy)
        for i, s in enumerate(samples):
            if fold_of[s.qid] == f:
                preds[i] = decide(s.signals, policy).answerable
    return CVResult(rule, [bool(p) for p in preds], policies)


def _by_qid(samples: Sequence[Sample]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.qid, []).append(i)
    return groups


def bootstrap(
    samples: Sequence[Sample],
    preds: Sequence[bool],
    baseline: Optional[Sequence[bool]] = None,
    n: int = 2000,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """按 qid 重采样的 95% 区间。

    给了 baseline 时同时算配对差（本规则 − 基线）的区间：两条规则在
    同一批重采样上比较，消掉样本本身的波动，比各自区间是否重叠更灵敏。
    """
    groups = list(_by_qid(samples).values())
    is_unans = np.array([s.label == UNANSWERABLE for s in samples])
    rejected = ~np.array(preds, dtype=bool)
    base_rejected = None if baseline is None else ~np.array(baseline, dtype=bool)

    rng = random.Random(seed)
    stats: Dict[str, List[float]] = {"recall": [], "frr": [], "d_recall": [], "d_frr": []}
    for _ in range(n):
        idx = np.array([i for g in rng.choices(groups, k=len(groups)) for i in g])
        u, a = is_unans[idx], ~is_unans[idx]
        r = rejected[idx]
        recall, frr = r[u].mean(), r[a].mean()
        stats["recall"].append(recall)
        stats["frr"].append(frr)
        if base_rejected is not None:
            b = base_rejected[idx]
            stats["d_recall"].append(recall - b[u].mean())
            stats["d_frr"].append(frr - b[a].mean())

    return {
        key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for key, v in stats.items() if v
    }
