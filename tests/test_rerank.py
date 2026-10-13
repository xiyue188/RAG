"""rerank 打分口径测试

守的是一个已经发生过的 bug：sentence-transformers 对单输出 cross-encoder
的 `predict()` 默认套 Sigmoid，`compute_relevance` 再套一次，得到
sigmoid(sigmoid(logit)) ∈ [0.5, 0.731]。开启 rerank 时 relevance 恒低于
0.75 的可答阈值，所有查询都被判为不可答。

不加载真实模型（约 1GB）：用替身模拟 predict 的默认行为 ——
不传 activation_fn 时返回概率，传 Identity 时返回 logit。
"""

from __future__ import annotations

import math

import pytest
import torch

from rag.retriever import Retriever


class FakeCrossEncoder:
    """模拟 sentence-transformers CrossEncoder.predict 对 num_labels=1 模型的行为。"""

    def __init__(self, logits):
        self.logits = logits
        self.activation_fn_seen = "not called"

    def predict(self, pairs, activation_fn=None):
        self.activation_fn_seen = activation_fn
        raw = torch.tensor(self.logits[: len(pairs)])
        fn = activation_fn if activation_fn is not None else torch.nn.Sigmoid()
        return fn(raw).tolist()


def _retriever_with(reranker) -> Retriever:
    retriever = Retriever.__new__(Retriever)  # 不走 __init__：会加载向量库与模型
    retriever._reranker = reranker
    return retriever


def _results(n):
    return [{"id": f"d{i}", "document": f"文档{i}", "cosine_distance": 0.4} for i in range(n)]


def test_rerank_requests_raw_logits():
    fake = FakeCrossEncoder([3.0, -3.0])
    _retriever_with(fake)._rerank_results("q", _results(2), top_k=2)
    assert isinstance(fake.activation_fn_seen, torch.nn.Identity)


def test_relevance_spans_full_probability_range():
    """logit ±3 → relevance ≈ 0.953 / 0.047。双重 sigmoid 下会是 0.722 / 0.512。"""
    fake = FakeCrossEncoder([3.0, -3.0])
    out = _retriever_with(fake)._rerank_results("q", _results(2), top_k=2)
    by_id = {r["id"]: r for r in out}
    assert by_id["d0"]["relevance"] == pytest.approx(1 / (1 + math.exp(-3)), abs=1e-4)
    assert by_id["d1"]["relevance"] == pytest.approx(1 / (1 + math.exp(3)), abs=1e-4)
    assert by_id["d0"]["rerank_logit"] == pytest.approx(3.0)


def test_rerank_order_by_logit():
    fake = FakeCrossEncoder([-1.0, 2.0, 0.5])
    out = _retriever_with(fake)._rerank_results("q", _results(3), top_k=3)
    assert [r["id"] for r in out] == ["d1", "d2", "d0"]
