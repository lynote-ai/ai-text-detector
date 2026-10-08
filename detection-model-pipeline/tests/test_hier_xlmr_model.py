"""HierXlmrForAiDetect 的前向与损失测试（tiny config，CPU 可跑，对齐 ModernBERT 版口径）。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_entity"))

from hier_xlmr_model import HierXlmrForAiDetect  # noqa: E402
from transformers import XLMRobertaConfig  # noqa: E402


def _tiny_config() -> XLMRobertaConfig:
    # XLMRobertaConfig 默认 pad_token_id=1 / bos=0 / eos=2，需小于 vocab_size。
    return XLMRobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )


def _tiny_model() -> HierXlmrForAiDetect:
    torch.manual_seed(42)
    model = HierXlmrForAiDetect(_tiny_config())
    model.eval()
    return model


def _batch(sents_per_doc: tuple[int, ...] = (3, 2)):
    input_ids = torch.randint(4, 100, (sum(sents_per_doc), 10))
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sents_per_doc)]
    )
    return input_ids, attention_mask, doc_ids


def test_xlmr_forward_shapes_and_total_loss():
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    labels_doc = torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    labels_origin = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] * 2 + [[0.3, 0.3, 0.4]])
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


def test_xlmr_alpha_weighting():
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
    assert torch.allclose(out2.loss, out2.loss_doc + 2.0 * sent_loss, atol=1e-6)


def test_xlmr_segment_pool_matches_naive_loop():
    model = _tiny_model()
    sent_vecs = torch.randn(5, 32)
    doc_ids = torch.tensor([0, 0, 0, 1, 1])
    pooled = model._segment_pool(sent_vecs, doc_ids, 2)
    for d in range(2):
        rows = sent_vecs[doc_ids == d]
        assert torch.allclose(pooled[d, :32], rows.mean(dim=0), atol=1e-6)
        assert torch.allclose(pooled[d, 32:], rows.max(dim=0).values, atol=1e-6)


def test_xlmr_fulltext_doc_channel_changes_logits():
    """V1 式全文通道生效性：不同全文输入产生不同 doc logits。"""
    model = _tiny_model()
    input_ids, attention_mask, doc_ids = _batch()
    kw = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        doc_ids=doc_ids, num_docs=2,
    )
    doc_a = torch.randint(4, 100, (2, 12))
    doc_b = torch.randint(4, 100, (2, 12))
    mask = torch.ones_like(doc_a)
    out_a = model(**kw, doc_input_ids=doc_a, doc_attention_mask=mask)
    out_b = model(**kw, doc_input_ids=doc_b, doc_attention_mask=mask)
    out_none = model(**kw)
    assert out_a.logits_doc.shape == (2, 3)
    assert not torch.allclose(out_a.logits_doc, out_b.logits_doc, atol=1e-6)
    # 不传全文时零填充退化路径可用。
    assert out_none.logits_doc.shape == (2, 3)
