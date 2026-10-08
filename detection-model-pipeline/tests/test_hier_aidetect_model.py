"""HierBertForAiDetect 的池化正确性与前向损失测试（tiny config，CPU 可跑）。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_entity"))

from hier_aidetect_model import HierBertForAiDetect  # noqa: E402
from transformers import ModernBertConfig  # noqa: E402


def _tiny_config() -> ModernBertConfig:
    # pad_token_id 必须显式覆盖：默认值 50283 超出 tiny vocab_size 会导致 Embedding 构造失败。
    return ModernBertConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )


def _tiny_model() -> HierBertForAiDetect:
    torch.manual_seed(42)
    model = HierBertForAiDetect(_tiny_config())
    model.eval()
    return model


def _batch(sents_per_doc: tuple[int, ...] = (3, 2)):
    input_ids = torch.randint(4, 100, (sum(sents_per_doc), 10))
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sents_per_doc)]
    )
    return input_ids, attention_mask, doc_ids


def test_segment_pool_matches_naive_loop():
    model = _tiny_model()
    sent_vecs = torch.randn(5, 32)
    doc_ids = torch.tensor([0, 0, 0, 1, 1])
    pooled = model._segment_pool(sent_vecs, doc_ids, 2)
    assert pooled.shape == (2, 64)
    for d in range(2):
        rows = sent_vecs[doc_ids == d]
        assert torch.allclose(pooled[d, :32], rows.mean(dim=0), atol=1e-6)
        assert torch.allclose(pooled[d, 32:], rows.max(dim=0).values, atol=1e-6)


def test_forward_shapes_and_total_loss():
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    labels_doc = torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    labels_origin = torch.rand(5, 3)
    labels_origin = labels_origin / labels_origin.sum(dim=1, keepdim=True)
    labels_human_ai = torch.rand(5, 1)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        doc_ids=doc_ids,
        num_docs=2,
        labels_doc=labels_doc,
        labels_origin=labels_origin,
        labels_human_ai=labels_human_ai,
    )
    assert out.logits_doc.shape == (2, 3)
    assert out.logits_origin.shape == (5, 3)
    assert out.logits_human_ai.shape == (5, 1)
    assert out.pooled.shape == (5, 32)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.loss_doc is not None and out.loss_origin is not None


def test_alpha_weighting():
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    kw = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        doc_ids=doc_ids,
        num_docs=2,
        labels_doc=torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]),
        labels_origin=torch.tensor([[1.0, 0.0, 0.0]] * 5),
    )
    out1 = model(**kw, alpha=1.0)
    out2 = model(**kw, alpha=2.0)
    sent_loss = out1.loss - out1.loss_doc
    expected = out2.loss_doc + 2.0 * sent_loss
    assert torch.allclose(out2.loss, expected, atol=1e-6)


def test_no_labels_gives_no_loss_and_doc_optional():
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
    )
    assert out.loss is None
    assert out.logits_doc is not None
    out_sent_only = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out_sent_only.logits_doc is None
    assert out_sent_only.logits_origin is not None


def test_forward_with_fulltext_doc_channel():
    """V1 式全文通道：doc 头输入 = [句池化 2H ‖ 全文编码 H]，None 时零填充退化。"""
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    doc_enc = torch.randint(4, 100, (2, 12))
    doc_mask = torch.ones_like(doc_enc)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
        doc_input_ids=doc_enc, doc_attention_mask=doc_mask,
        labels_doc=torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]),
    )
    assert out.logits_doc.shape == (2, 3)
    assert out.loss is not None and torch.isfinite(out.loss)

    # 不传全文输入时走零填充退化路径，形状不变。
    out_no_full = model(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
    )
    assert out_no_full.logits_doc.shape == (2, 3)

    # 全文输入不同应导致 doc logits 不同（证明通道真实生效，非摆设）。
    doc_enc2 = torch.randint(4, 100, (2, 12))
    out_full2 = model(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
        doc_input_ids=doc_enc2, doc_attention_mask=doc_mask,
    )
    assert not torch.allclose(out.logits_doc, out_full2.logits_doc, atol=1e-6)


def test_cascade_sent_stats_change_doc_logits():
    """级联融合：句级概率分布不同（backbone 输出不同）应改变 doc logits。"""
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    kw = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
    )
    out1 = model(**kw)
    out2 = model(
        input_ids=torch.randint(4, 100, (5, 10)), attention_mask=torch.ones(5, 10),
        doc_ids=torch.tensor([0, 0, 0, 1, 1]), num_docs=2,
    )
    # 句子输入不同 → 句向量不同 → 级联统计不同 → doc logits 不同。
    assert not torch.allclose(out1.logits_doc, out2.logits_doc, atol=1e-6)


def test_class_weighted_soft_ce():
    """类权重：None 等价标准软 CE；少数类样本在 batch 内 loss 占比放大。"""
    model = _tiny_model()
    # 2 个少数类（第 3 类）样本错判严重 + 1 个多数类样本判对。
    logits = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.05, 0.0]])
    soft = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    w_none = model._weighted_soft_ce(logits, soft, None)
    w_heavy = model._weighted_soft_ce(logits, soft, [1.0, 1.0, 2.0])
    assert torch.isfinite(w_none) and torch.isfinite(w_heavy)
    # 少数类样本（高 loss）权重 ×2 → 其在加权平均中占比 2/3 → 4/5，总 loss 变大。
    assert w_heavy > w_none
    # None 路径与标准软 CE 数值一致。
    ref = torch.nn.CrossEntropyLoss()(logits, soft)
    assert torch.allclose(w_none, ref, atol=1e-6)


def test_class_weights_from_config():
    config = _tiny_config()
    config.sent_class_weights = [1.0, 1.0, 2.0]
    config.doc_class_weights = [1.0, 1.0, 2.0]
    torch.manual_seed(7)
    model = HierBertForAiDetect(config)
    model.eval()
    input_ids, attention_mask, doc_ids = _batch()
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
        labels_doc=torch.tensor([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]),
        labels_origin=torch.tensor([[1.0, 0.0, 0.0]] * 5),
    )
    assert out.loss is not None and torch.isfinite(out.loss)
