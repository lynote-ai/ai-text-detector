"""DynamicBatcher 攒批器测试：拼批 forward 结果与逐篇 forward 数值一致。"""

import sys
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_entity"))

from m_serve_hier_api import DynamicBatcher, _softmax_2d  # noqa: E402


def _tiny_model_and_tokenizer():
    from hier_xlmr_model import HierXlmrForAiDetect
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig

    words = ["hello", "world", "ai", "human", "text", "cat", "dog", "runs", "fast", "slow"]
    vocab = {"[PAD]": 0, "[UNK]": 1} | {w: i + 2 for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = XLMRobertaConfig(
        vocab_size=20, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=512,
    )
    torch.manual_seed(42)
    model = HierXlmrForAiDetect(config)
    model.eval()
    return model, tokenizer


def _make_job(text, sentence_texts):
    return {
        "text": text,
        "sentences": [{"text": s} for s in sentence_texts],
        "event": threading.Event(),
        "result": None,
        "error": None,
    }


def test_batched_results_match_separate_forward():
    """两篇拼批的输出概率与各自单独 forward 完全一致。"""
    model, tokenizer = _tiny_model_and_tokenizer()
    device = torch.device("cpu")
    batcher = DynamicBatcher(model, tokenizer, device)

    job1 = _make_job("hello world ai human text", ["hello world", "cat runs fast"])
    job2 = _make_job("ai text hello", ["ai text", "dog runs slow", "hello cat"])

    # 逐篇单独 forward 的参考结果。
    refs = []
    for job in (job1, job2):
        sent_texts = [s["text"] for s in job["sentences"]]
        enc = tokenizer(sent_texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        doc_enc = tokenizer([job["text"]], padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                doc_ids=torch.zeros(len(sent_texts), dtype=torch.long),
                num_docs=1,
                doc_input_ids=doc_enc["input_ids"],
                doc_attention_mask=doc_enc["attention_mask"],
            )
        refs.append((
            _softmax_2d(out.logits_origin.numpy()),
            out.logits_doc.numpy()[0],
        ))

    # 拼批处理两个 job。
    batcher.process_jobs([job1, job2])
    assert job1["error"] is None and job2["error"] is None
    assert job1["event"].is_set() and job2["event"].is_set()

    for job, (ref_sent, ref_doc) in zip((job1, job2), refs):
        np.testing.assert_allclose(job["result"]["sent_probs"], ref_sent, atol=1e-5)
        np.testing.assert_allclose(job["result"]["doc_logits"], ref_doc, atol=1e-5)


def test_batcher_error_propagates_to_all_jobs():
    """forward 异常时，同批所有 job 收到 error 而非悬挂。"""
    model, tokenizer = _tiny_model_and_tokenizer()
    batcher = DynamicBatcher(model, tokenizer, torch.device("cpu"))

    class Boom:
        def __call__(self, **_kw):
            raise RuntimeError("forward failed")

    batcher.model = Boom()
    job1 = _make_job("hello world", ["hello world"])
    job2 = _make_job("ai text", ["ai text"])
    batcher.process_jobs([job1, job2])
    assert isinstance(job1["error"], RuntimeError)
    assert isinstance(job2["error"], RuntimeError)
    assert job1["event"].is_set() and job2["event"].is_set()


def test_submit_via_background_thread():
    """生产路径集成：start() 起后台线程，submit 阻塞等待并返回结果。"""
    model, tokenizer = _tiny_model_and_tokenizer()
    batcher = DynamicBatcher(model, tokenizer, torch.device("cpu"), window_s=0.05)
    batcher.start()

    from m_serve_hier_api import _softmax_2d as softmax

    job = _make_job("hello world ai", ["hello world", "ai text"])
    enc = tokenizer([s["text"] for s in job["sentences"]], padding=True,
                    return_tensors="pt")
    doc_enc = tokenizer([job["text"]], return_tensors="pt")
    with torch.no_grad():
        ref = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            doc_ids=torch.zeros(2, dtype=torch.long), num_docs=1,
            doc_input_ids=doc_enc["input_ids"],
            doc_attention_mask=doc_enc["attention_mask"],
        )

    result = batcher.submit(job["text"], job["sentences"])
    np.testing.assert_allclose(
        result["sent_probs"], softmax(ref.logits_origin.numpy()), atol=1e-5
    )
    assert result["doc_logits"].shape == (3,)
