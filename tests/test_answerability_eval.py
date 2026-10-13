"""rag/answerability.py 与 rag/eval/answerability.py 的测试

重点是两处"静默出错"的风险：

1. 无信号的 0.0 被当成真实分数 —— 会让分差、背景分全部失真
2. 评测选参用的向量化判定与生产 `decide` 不一致 —— 选参阶段和计分阶段
   在执行两套规则，量出来的数字没有意义（`TestVectorizedMatchesDecide`）
"""

from __future__ import annotations

import random

import pytest

from rag.answerability import (
    BASIS_COSINE,
    BASIS_NONE,
    BASIS_RERANK,
    REASON_GAP12_LOW,
    REASON_GAP_BG_LOW,
    REASON_NO_RESULTS,
    REASON_NO_SIGNAL,
    REASON_OK,
    REASON_TOP1_LOW,
    AnswerabilityPolicy,
    assess,
    decide,
    extract_signals,
    scores_from_results,
)
from rag.eval.answerability import (
    ANSWERABLE,
    UNANSWERABLE,
    RULES,
    Sample,
    _Matrix,
    bootstrap,
    build_samples,
    cross_validate,
    predict,
    rates,
    select_policy,
)
from rag.eval.runner import QueryResult


# ============================================================
# extract_signals
# ============================================================

class TestExtractSignals:
    def test_empty(self):
        s = extract_signals([], BASIS_COSINE)
        assert s.n_scored == 0 and s.top1 is None and s.gap12 is None

    def test_single_has_no_gaps(self):
        s = extract_signals([0.8], BASIS_COSINE)
        assert s.top1 == 0.8 and s.top2 is None and s.gap12 is None and s.gap_bg is None

    def test_order_independent_and_values(self):
        s = extract_signals([0.70, 0.90, 0.80, 0.60], BASIS_COSINE)
        assert s.top1 == 0.90 and s.top2 == 0.80
        assert s.gap12 == pytest.approx(0.10)
        # 背景 = 除 top1 外的中位数 = median(0.8, 0.7, 0.6)
        assert s.background == pytest.approx(0.70)
        assert s.gap_bg == pytest.approx(0.20)

    def test_ties(self):
        s = extract_signals([0.8, 0.8, 0.8], BASIS_COSINE)
        assert s.gap12 == 0.0 and s.gap_bg == 0.0


# ============================================================
# scores_from_results：口径与无信号
# ============================================================

class TestScoresFromResults:
    def test_cosine(self):
        scores, basis = scores_from_results([{"cosine_distance": 0.4}, {"cosine_distance": 0.6}])
        assert basis == BASIS_COSINE
        assert scores == pytest.approx([0.8, 0.7])

    def test_bm25_only_entries_are_excluded_not_zero(self):
        """纯 BM25 命中没有相关性，不能以 0.0 进入特征计算。"""
        scores, basis = scores_from_results([{"cosine_distance": 0.4}, {"bm25_score": 12.3}])
        assert basis == BASIS_COSINE and scores == pytest.approx([0.8])

    def test_rerank_preferred_and_not_mixed(self):
        """有 rerank 分数时只取 rerank 分数：两种口径不可比。"""
        results = [
            {"cosine_distance": 0.2, "rerank_logit": 0.0},   # sigmoid(0)=0.5
            {"cosine_distance": 0.1},                         # 只有余弦，不得混入
        ]
        scores, basis = scores_from_results(results)
        assert basis == BASIS_RERANK and scores == pytest.approx([0.5])

    def test_no_signal(self):
        assert scores_from_results([{"bm25_score": 3.0}]) == ([], BASIS_NONE)

    def test_bool_is_not_a_score(self):
        scores, basis = scores_from_results([{"rerank_logit": True, "cosine_distance": 0.4}])
        assert basis == BASIS_COSINE


# ============================================================
# decide
# ============================================================

def _sig(*scores, basis=BASIS_COSINE):
    return extract_signals(scores, basis)


class TestDecide:
    def test_no_signal_trusts_ranking(self):
        """与 M0 以来的行为一致：纯 BM25 不做阈值判断。"""
        d = decide(extract_signals([], BASIS_NONE), AnswerabilityPolicy(min_top1=0.99))
        assert d.answerable and d.reason == REASON_NO_SIGNAL

    def test_no_results(self):
        d = decide(_sig(), AnswerabilityPolicy())
        assert not d.answerable and d.reason == REASON_NO_RESULTS

    def test_top1_boundary_inclusive(self):
        assert decide(_sig(0.75, 0.5), AnswerabilityPolicy(min_top1=0.75)).answerable
        d = decide(_sig(0.7499, 0.5), AnswerabilityPolicy(min_top1=0.75))
        assert not d.answerable and d.reason == REASON_TOP1_LOW
        assert "0.750" in d.explanation  # 拒答说得出具体数值

    def test_gap12(self):
        d = decide(_sig(0.80, 0.79), AnswerabilityPolicy(min_gap12=0.05))
        assert not d.answerable and d.reason == REASON_GAP12_LOW

    def test_gap_bg(self):
        d = decide(_sig(0.80, 0.79, 0.78), AnswerabilityPolicy(min_gap_bg=0.05))
        assert not d.answerable and d.reason == REASON_GAP_BG_LOW

    def test_unevaluable_gap_does_not_reject(self):
        """只有一条结果时分差不可算 —— 缺少证据不构成拒答理由。"""
        d = decide(_sig(0.9), AnswerabilityPolicy(min_gap12=0.05, min_gap_bg=0.05))
        assert d.answerable and d.reason == REASON_OK

    def test_conditions_are_anded(self):
        policy = AnswerabilityPolicy(min_top1=0.7, min_gap12=0.05)
        assert decide(_sig(0.9, 0.8), policy).answerable
        assert not decide(_sig(0.9, 0.88), policy).answerable
        assert not decide(_sig(0.6, 0.4), policy).answerable

    def test_assess_end_to_end(self):
        d = assess([{"cosine_distance": 0.6}], AnswerabilityPolicy(min_top1=0.75))
        assert not d.answerable and d.signals.top1 == pytest.approx(0.7)
        assert assess([], AnswerabilityPolicy()).reason == REASON_NO_RESULTS


# ============================================================
# build_samples
# ============================================================

def _qr(qid, ids, rels, gold):
    return QueryResult(qid, "q", "rrf", ids, rels, gold, 1.0, {})


class TestBuildSamples:
    def test_pair_and_gold_removal(self):
        r = _qr("1", ["g1", "x1", "g2", "x2"], [0.9, 0.8, 0.85, 0.7], ["g1", "g2"])
        ans, unans = build_samples([r], k=20)
        assert (ans.label, unans.label) == (ANSWERABLE, UNANSWERABLE)
        assert ans.signals.top1 == 0.9 and ans.n_gold == 2
        assert unans.signals.top1 == 0.8 and unans.signals.n_scored == 2

    def test_zero_is_missing(self):
        """RRF 里纯 BM25 命中的 0.0 不进入特征。"""
        r = _qr("1", ["a", "b", "c"], [0.9, 0.0, 0.7], [])
        ans = build_samples([r], k=20)[0]
        assert ans.signals.n_scored == 2 and ans.signals.top2 == 0.7

    def test_visibility_truncates_before_filtering(self):
        """前 k 条按检索排名截取，再去掉无信号条目 —— 与生产看到的一致。"""
        r = _qr("1", ["a", "b", "c", "d"], [0.9, 0.0, 0.0, 0.8], [])
        ans = build_samples([r], k=3)[0]
        assert ans.signals.n_scored == 1  # 第 4 条在可见范围外

    def test_all_missing_in_view_is_no_signal(self):
        r = _qr("1", ["a", "b"], [0.0, 0.0], [])
        assert build_samples([r], k=3)[0].signals.basis == BASIS_NONE

    def test_all_gold_yields_no_unanswerable(self):
        r = _qr("1", ["g1", "g2"], [0.9, 0.8], ["g1", "g2"])
        assert [s.label for s in build_samples([r], k=20)] == [ANSWERABLE]

    def test_errors_skipped(self):
        r = QueryResult("1", "q", "rrf", [], [], [], 1.0, {}, error="boom")
        assert build_samples([r], k=3) == []


# ============================================================
# 选参与计分
# ============================================================

def _synthetic(n=60, seed=1):
    """可分的合成数据：有答案 top1 高且领先明显，无答案平坦。"""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        top = rng.uniform(0.80, 0.90)
        samples.append(Sample(str(i), ANSWERABLE, _sig(top, top - 0.06, top - 0.08, top - 0.09), 3))
        flat = rng.uniform(0.80, 0.90)  # 与有答案同分布：top1 完全不可分
        samples.append(Sample(str(i), UNANSWERABLE, _sig(flat, flat - 0.005, flat - 0.01), 3))
    return samples


class TestVectorizedMatchesDecide:
    """选参用的 numpy 判定必须与生产 decide 逐条一致。"""

    def test_random_policies(self):
        rng = random.Random(7)
        samples = _synthetic() + [
            Sample("s", ANSWERABLE, _sig(0.9), 1),                         # 分差不可算
            Sample("n", ANSWERABLE, extract_signals([], BASIS_NONE), 0),   # 无信号
            Sample("e", UNANSWERABLE, _sig(), 0),                          # 无结果
        ]
        m = _Matrix(samples)
        for _ in range(200):
            policy = AnswerabilityPolicy(
                min_top1=rng.choice([None, rng.uniform(0.6, 0.95)]),
                min_gap12=rng.choice([None, rng.uniform(0.0, 0.1)]),
                min_gap_bg=rng.choice([None, rng.uniform(0.0, 0.1)]),
            )
            assert list(m.predict(policy)) == predict(samples, policy), policy


class TestSelectAndCV:
    def test_select_respects_frr_constraint(self):
        samples = _synthetic()
        for rule in RULES:
            policy = select_policy(samples, rule, max_frr=0.10)
            _, frr = rates(samples, predict(samples, policy))
            assert frr <= 0.10, rule

    def test_relative_feature_beats_absolute_on_flat_negatives(self):
        """合成数据上两类 top1 重叠而分差不重叠 —— gap 规则应胜出。

        这是对评测框架的检验（能识别出真正有效的特征），不是对真实数据的断言。
        """
        samples = _synthetic()
        r0 = rates(samples, cross_validate(samples, "R0 top1").preds)[0]
        r1 = rates(samples, cross_validate(samples, "R1 top1+gap12").preds)[0]
        assert r1 > r0 + 0.5

    def test_cv_covers_every_sample_once_and_is_deterministic(self):
        samples = _synthetic()
        a = cross_validate(samples, "R2 gap_bg", seed=3)
        b = cross_validate(samples, "R2 gap_bg", seed=3)
        assert len(a.preds) == len(samples) and a.preds == b.preds
        assert len(a.policies) == 5


class TestBootstrap:
    def test_interval_contains_point_estimate(self):
        samples = _synthetic()
        preds = cross_validate(samples, "R1 top1+gap12").preds
        recall, frr = rates(samples, preds)
        ci = bootstrap(samples, preds, n=300)
        assert ci["recall"][0] <= recall <= ci["recall"][1]
        assert ci["frr"][0] <= frr <= ci["frr"][1]

    def test_paired_difference_with_itself_is_zero(self):
        samples = _synthetic()
        preds = cross_validate(samples, "R0 top1").preds
        ci = bootstrap(samples, preds, baseline=preds, n=200)
        assert ci["d_recall"] == (0.0, 0.0) and ci["d_frr"] == (0.0, 0.0)


class TestLegacyRerankRecords:
    def test_undo_double_sigmoid_is_exact_inverse(self):
        import math
        from rag.eval.answerability import undo_double_sigmoid

        probs = [0.001, 0.3, 0.786, 0.999]
        stored = [1 / (1 + math.exp(-p)) for p in probs]  # 修复前的双重 sigmoid
        r = _qr("1", ["a", "b", "c", "d"], stored, [])
        assert undo_double_sigmoid([r])[0].relevances == pytest.approx(probs, abs=1e-6)

    def test_rerank_basis_keeps_near_zero_scores(self):
        """rerank 口径不做 0.0 ⇒ 缺失：概率可以合法地接近 0。"""
        r = _qr("1", ["a", "b"], [0.9, 0.0], [])
        assert build_samples([r], k=20, basis=BASIS_RERANK)[0].signals.n_scored == 2
