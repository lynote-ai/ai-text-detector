"""双级输出映射与服务接口测试（monkeypatch 推理函数，不起真权重）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from m_serve_hier_api import (  # noqa: E402
    DOC_CLASS_ENUM,
    build_gptzero_document,
    build_gptzero_response,
    classify_confidence,
)

SENT_PROBS = np.array([
    [0.05, 0.90, 0.05],
    [0.80, 0.10, 0.10],
    [0.10, 0.20, 0.70],
])


def test_document_schema_matches_gptzero_fields():
    paragraphs_meta = [
        {"start_sentence_index": 0, "num_sentences": 2},
        {"start_sentence_index": 2, "num_sentences": 1},
    ]
    sentences = [{"text": "Ai sentence one."}, {"text": "Human two."},
                 {"text": "Polished three."}]
    doc = build_gptzero_document(
        text="Ai sentence one. Human two.\n\nPolished three.",
        paragraphs_meta=paragraphs_meta,
        sentences=sentences,
        sent_origin_probs=SENT_PROBS,
        doc_probs=np.array([0.05, 0.85, 0.10]),
        language="en",
        document_id="doc-1",
    )
    expected_keys = {
        "paragraphs", "sentences", "classProbabilities", "confidenceThresholdsRaw",
        "confidenceScoresRaw", "subclass", "pageNumber", "language", "inputText",
        "documentId", "predictedClass", "confidenceScore", "confidenceCategory",
        "documentClassification", "resultMessage", "completelyGeneratedProb",
        "averageGeneratedProb", "overallBurstiness", "writingStats", "version",
        "neatVersion",
    }
    assert expected_keys <= set(doc.keys())
    assert doc["predictedClass"] == "ai"
    assert doc["documentClassification"] == DOC_CLASS_ENUM["ai"] == "AI_ONLY"
    assert doc["confidenceCategory"] == "high"  # max prob 0.85
    # completelyGeneratedProb = P(ai)（实测口径）。
    assert doc["completelyGeneratedProb"] == pytest.approx(0.85)
    # averageGeneratedProb = 句子 generatedProb 均值 = mean(P(ai)+P(paraphrased))。
    gen = SENT_PROBS[:, 1] + SENT_PROBS[:, 2]
    assert doc["averageGeneratedProb"] == pytest.approx(float(gen.mean()))
    assert doc["overallBurstiness"] == 0 and doc["writingStats"] == {}
    s0, s1, s2 = doc["sentences"]
    assert s0["generatedProb"] == pytest.approx(0.95) and s0["highlightSentenceForAi"]
    assert not s1["highlightSentenceForAi"]
    assert s2["specialHighlightType"] == "polished"
    assert s2["classProbabilities"]["paraphrased"] == pytest.approx(0.70)
    p0, p1 = doc["paragraphs"]
    assert p0["startSentenceIndex"] == 0 and p0["numSentences"] == 2
    assert p0["completelyGeneratedProb"] == pytest.approx(float(gen[:2].mean()))
    assert p1["completelyGeneratedProb"] == pytest.approx(float(gen[2]))


def test_classify_confidence_thresholds():
    assert classify_confidence(0.9) == "high"
    assert classify_confidence(0.8) == "high"
    assert classify_confidence(0.79) == "medium"
    assert classify_confidence(0.6) == "medium"
    assert classify_confidence(0.59) == "low"
    assert classify_confidence(0.33) == "low"
    assert classify_confidence(0.32) == "reject"


def test_response_top_level_structure():
    doc = {"documentId": "d1", "predictedClass": "human"}
    resp = build_gptzero_response([doc])
    assert set(resp.keys()) >= {"version", "neatVersion", "scanId", "documents"}
    assert len(resp["scanId"]) == 32 and resp["documents"] == [doc]


def test_api_endpoints_with_fake_model(monkeypatch):
    from fastapi.testclient import TestClient

    import m_serve_hier_api as serve
    from m_serve_hier_api import build_gptzero_document

    def fake_detect_texts(texts, *_args, **_kwargs):
        documents = []
        for text in texts:
            if not text.strip():
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=400, detail="text contains no parseable sentence"
                )
            paragraphs_meta = [{"start_sentence_index": 0, "num_sentences": 1}]
            sentences = [{"text": "One sentence."}]
            sent_probs = np.array([[0.9, 0.08, 0.02]])
            doc = build_gptzero_document(
                text=text, paragraphs_meta=paragraphs_meta, sentences=sentences,
                sent_origin_probs=sent_probs, doc_probs=np.array([0.92, 0.06, 0.02]),
                language="en", document_id="fake-doc",
            )
            documents.append(doc)
        return documents

    monkeypatch.setattr(serve, "detect_documents", fake_detect_texts)
    monkeypatch.setattr(serve, "_MODEL_LOADED", True)
    client = TestClient(serve.app)

    # 探针也走统一包装。
    probe = client.get("/")
    assert probe.status_code == 200
    assert probe.json()["code"] == 0
    assert probe.json()["data"]["model_loaded"] is True

    # 成功：单字符串输入。
    r = client.post("/detect", json={"text": "One sentence."})
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0 and body["msg"] == "success"
    assert len(body["data"]["documents"]) == 1
    assert body["data"]["documents"][0]["predictedClass"] == "human"

    # 成功：数组输入，一个接口直接批量。
    r_arr = client.post("/detect", json={"text": ["One sentence."] * 3})
    assert r_arr.status_code == 200
    assert len(r_arr.json()["data"]["documents"]) == 3

    # 兼容端点 /detect/batch 行为不变。
    r2 = client.post("/detect/batch", json={"texts": ["One sentence."] * 3})
    assert r2.status_code == 200 and len(r2.json()["data"]["documents"]) == 3

    # 失败：HTTP 码保留 + 业务 code + data=null。
    r3 = client.post("/detect", json={"text": ""})
    assert r3.status_code == 422
    assert r3.json()["code"] == 1001 and r3.json()["data"] is None

    r4 = client.post("/detect", json={"text": ["x"] * 33})
    assert r4.status_code == 413
    assert r4.json()["code"] == 1003

    r5 = client.post("/detect", json={"text": "   "})
    assert r5.status_code == 400
    assert r5.json()["code"] == 1002 and r5.json()["data"] is None

    # 空数组输入被校验拦截。
    r6 = client.post("/detect", json={"text": []})
    assert r6.status_code == 422
