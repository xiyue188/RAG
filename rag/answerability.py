"""可答性判断 —— 检索到的证据是否足以支撑基于文档的回答

## 为什么单独成模块

M1 证明了单一阈值不可行：有答案与无答案查询的 top1 relevance 分布严重
重叠，误拒率 ≤10% 时只能识别 19.1% 的无答案查询。重做需要尝试多种特征与
规则，而评测与生产必须调用**同一份**判断代码 —— 评测里另写一份，量出来的
就不是生产的行为。

## 两个函数分开

- `extract_signals`：从分数序列算特征。这是**事实**，与采用哪条规则无关。
- `decide`：按策略（`AnswerabilityPolicy`）判定。这是**策略**。

评测对同一组特征试多种策略、生产只用一种。合在一起，评测就得把特征提取
抄一遍。

## 无信号不是 0 分

纯 BM25 命中没有 [0,1] 的相关性（见 `scoring.has_relevance_signal`），
`compute_relevance` 对它返回 0.0。这个 0.0 是"缺失"，不是"极不相关"。
特征只用有信号的条目计算；把缺失当 0 分会让 top2、背景分全部失真。

## 不混用口径

relevance 有两种物理含义：余弦 `1 - d/2` 与 rerank `sigmoid(logit)`。
两者尺度不同（M1：同为单阈值法，余弦口径最优阈值 0.75，rerank 口径 0.65），
同一序列里混用会让分差失去意义。有 rerank 分数时只用 rerank 分数。

本模块不 import `backend`，不做 I/O。
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .scoring import compute_relevance, has_relevance_signal

BASIS_COSINE = "cosine"
BASIS_RERANK = "rerank"
BASIS_NONE = "none"

# 判定理由的稳定编码。用于 runs 表聚合与前端分支，改名即破坏历史数据的可比性。
REASON_OK = "ok"
REASON_NO_RESULTS = "no_results"
REASON_NO_SIGNAL = "no_signal"
REASON_TOP1_LOW = "top1_below_min"
REASON_GAP12_LOW = "gap12_below_min"
REASON_GAP_BG_LOW = "gap_bg_below_min"


@dataclass(frozen=True)
class AnswerabilitySignals:
    """从一组检索分数中提取的特征。分数缺失时对应字段为 None。"""

    basis: str
    # 有相关性信号的条目数（无信号条目不计入）
    n_scored: int
    top1: Optional[float]
    top2: Optional[float]
    # top1 - top2：第一名是否明显领先第二名
    gap12: Optional[float]
    # 背景分：除 top1 外其余分数的中位数
    background: Optional[float]
    # top1 - 背景分：第一名比"普通文档"高出多少。
    # 动机是向量的各向异性：任意两段中文的余弦都挤在窄的高位区间，
    # 绝对值几乎不携带信息，相对值可能才有
    gap_bg: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerabilityPolicy:
    """判定策略。字段为 None 表示不使用该条件。

    多个条件之间是"且"：任一不满足即判为不可答。
    """

    min_top1: Optional[float] = None
    min_gap12: Optional[float] = None
    min_gap_bg: Optional[float] = None


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerable: bool
    # 稳定编码，见模块顶部 REASON_*
    reason: str
    # 人可读说明，含具体数值。拒答时要说得出为什么（对齐"禁止无声降级"）
    explanation: str
    signals: AnswerabilitySignals


def extract_signals(scores: Iterable[float], basis: str) -> AnswerabilitySignals:
    """从有信号的分数序列提取特征。

    参数:
        scores: 只含有相关性信号的分数，顺序任意（内部按降序排）
        basis: 分数口径，BASIS_COSINE / BASIS_RERANK
    """
    ordered = sorted((float(s) for s in scores), reverse=True)
    n = len(ordered)
    if n == 0:
        return AnswerabilitySignals(basis, 0, None, None, None, None, None)

    top1 = ordered[0]
    if n == 1:
        return AnswerabilitySignals(basis, 1, top1, None, None, None, None)

    top2 = ordered[1]
    background = statistics.median(ordered[1:])
    return AnswerabilitySignals(
        basis=basis,
        n_scored=n,
        top1=top1,
        top2=top2,
        gap12=top1 - top2,
        background=background,
        gap_bg=top1 - background,
    )


def scores_from_results(results: Sequence[Dict[str, Any]]) -> Tuple[List[float], str]:
    """从检索结果字典取出有信号的分数及其口径。

    有任一条带 rerank 分数时只取 rerank 分数：精排是对同一批候选的更准判断，
    与余弦混在一起排序没有意义。
    """
    reranked = [
        r for r in results
        if isinstance(r.get("rerank_logit"), (int, float))
        and not isinstance(r.get("rerank_logit"), bool)
    ]
    if reranked:
        return [compute_relevance(r) for r in reranked], BASIS_RERANK

    scored = [r for r in results if has_relevance_signal(r)]
    if scored:
        return [compute_relevance(r) for r in scored], BASIS_COSINE
    return [], BASIS_NONE


def decide(signals: AnswerabilitySignals, policy: AnswerabilityPolicy) -> AnswerabilityDecision:
    """按策略判定可答性。

    无信号时判为可答：纯 BM25 检索没有可比的相关性，阈值判断会拒绝全部查询。
    此时退化为信任检索层的排序 —— 与 M0 以来 `assess_context` 的行为一致。
    无结果时判为不可答。
    """
    if signals.basis == BASIS_NONE:
        return AnswerabilityDecision(
            True, REASON_NO_SIGNAL, "检索结果没有相关性分数，按检索排序采用", signals
        )
    if signals.n_scored == 0 or signals.top1 is None:
        return AnswerabilityDecision(False, REASON_NO_RESULTS, "没有检索到任何结果", signals)

    if policy.min_top1 is not None and signals.top1 < policy.min_top1:
        return AnswerabilityDecision(
            False, REASON_TOP1_LOW,
            f"最相关片段的相关性 {signals.top1:.3f} 低于下限 {policy.min_top1:.3f}",
            signals,
        )

    # 只有一条结果时没有分差可算。按"条件不可评估则不据此拒答"处理：
    # 拒答需要证据，缺少证据不构成拒答理由
    if (
        policy.min_gap12 is not None
        and signals.gap12 is not None
        and signals.gap12 < policy.min_gap12
    ):
        return AnswerabilityDecision(
            False, REASON_GAP12_LOW,
            f"第一名仅领先第二名 {signals.gap12:.3f}，低于下限 {policy.min_gap12:.3f}",
            signals,
        )
    if (
        policy.min_gap_bg is not None
        and signals.gap_bg is not None
        and signals.gap_bg < policy.min_gap_bg
    ):
        return AnswerabilityDecision(
            False, REASON_GAP_BG_LOW,
            f"最相关片段仅比其余片段高 {signals.gap_bg:.3f}，低于下限 {policy.min_gap_bg:.3f}",
            signals,
        )

    return AnswerabilityDecision(True, REASON_OK, "证据充分", signals)


def assess(results: Sequence[Dict[str, Any]], policy: AnswerabilityPolicy) -> AnswerabilityDecision:
    """生产入口：检索结果 → 判定。"""
    if not results:
        return AnswerabilityDecision(
            False, REASON_NO_RESULTS, "没有检索到任何结果",
            AnswerabilitySignals(BASIS_COSINE, 0, None, None, None, None, None),
        )
    scores, basis = scores_from_results(results)
    return decide(extract_signals(scores, basis), policy)
