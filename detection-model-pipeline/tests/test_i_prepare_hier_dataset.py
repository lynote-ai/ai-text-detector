"""i_prepare_hier_dataset 的排除/去重/抽取逻辑测试（小 fixture 驱动）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from i_prepare_hier_dataset import (  # noqa: E402
    extract_sample,
    load_excluded_scan_ids,
    md5_norm_text,
)


def _gptzero_response(doc_class: str = "human", language: str = "en") -> str:
    return json.dumps({
        "documents": [{
            "predictedClass": doc_class,
            "language": language,
            "classProbabilities": {"human": 0.9, "ai": 0.08, "mixed": 0.02},
            "sentences": [{
                "sentence": "Hello world.",
                "classProbabilities": {"human": 0.95, "ai": 0.04, "paraphrased": 0.01},
            }],
        }]
    })


def _csv_row(scan_id: str, text: str, doc_class: str = "human") -> dict:
    return {
        "scan_id": scan_id,
        "request_payload": json.dumps({"document": text}),
        "response_payload": _gptzero_response(doc_class),
        "word_count": str(len(text.split())),
    }


LONG_TEXT = ("This is a long enough sentence with ten words at least here. " * 3).strip()


def test_md5_norm_text_stable_across_whitespace():
    assert md5_norm_text("a  b\n\nc") == md5_norm_text("a b c")


def test_load_excluded_scan_ids(tmp_path):
    p = tmp_path / "old.jsonl"
    p.write_text('{"scan_id": "s1"}\n{"scan_id": "s2"}\n\n', encoding="utf-8")
    assert load_excluded_scan_ids([str(p)]) == {"s1", "s2"}


def test_extract_sample_fields():
    row = _csv_row("abc", LONG_TEXT, "mixed")
    s = extract_sample(row)
    assert s is not None
    assert s["scan_id"] == "abc"
    assert s["doc_label"] == [0.9, 0.08, 0.02]
    assert s["sentences"][0]["text"] == "Hello world."
    assert s["sentences"][0]["y_origin"] == [0.95, 0.04, 0.01]
    assert s["sentences"][0]["y_human_ai"] == 0.95
    assert s["gptzero_doc_class"] == "mixed"
    assert s["gptzero_language"] == "en"


def test_extract_sample_rejects_short_and_empty():
    short = _csv_row("x1", "too short")
    assert extract_sample(short) is None
    empty_resp = _csv_row("x2", LONG_TEXT)
    empty_resp["response_payload"] = json.dumps({"documents": [{"sentences": []}]})
    assert extract_sample(empty_resp) is None


def test_extract_sample_truncates_256_sentences():
    resp = json.dumps({"documents": [{
        "predictedClass": "ai", "language": "en",
        "classProbabilities": {"human": 0.1, "ai": 0.8, "mixed": 0.1},
        "sentences": [{
            "sentence": f"Sentence number {i}.",
            "classProbabilities": {"human": 0.1, "ai": 0.8, "paraphrased": 0.1},
        } for i in range(300)],
    }]})
    row = {
        "scan_id": "big", "word_count": "3000",
        "request_payload": json.dumps({"document": "word " * 3000}),
        "response_payload": resp,
    }
    assert len(extract_sample(row)["sentences"]) == 256


def _dist_sample(doc_label, sent_origins, lang="en", text="w " * 50):
    return {
        "scan_id": "x", "text": text, "gptzero_language": lang,
        "doc_label": doc_label,
        "sentences": [
            {"text": "s", "y_origin": y, "y_human_ai": y[0]} for y in sent_origins
        ],
    }


def test_accumulate_dist_stats_counts_and_lengths():
    from i_prepare_hier_dataset import new_dist_stats, accumulate_dist_stats

    stats = new_dist_stats()
    accumulate_dist_stats(stats, _dist_sample(
        [0.1, 0.8, 0.1], [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]], lang="es",
    ))
    accumulate_dist_stats(stats, _dist_sample(
        [0.97, 0.02, 0.01], [[0.99, 0.005, 0.005]], lang="en",
    ))
    assert stats["doc_class"] == {"ai": 1, "human": 1}
    # 3 句：[0.9,..]=human、[0.2,0.7,..]=ai、[0.99,..]=human。
    assert stats["sent_class"] == {"human": 2, "ai": 1}
    assert stats["lang"] == {"es": 1, "en": 1}
    assert stats["n_docs"] == 2 and stats["n_sents"] == 3
    # 词数 50/句数 2 与 50/1。
    assert stats["word_counts"] == [50, 50]
    assert stats["sent_counts"] == [2, 1]
    # doc 标签 max 概率：0.8 与 0.97。
    assert stats["doc_label_max"] == [0.8, 0.97]
