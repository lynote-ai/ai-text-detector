"""AI 文本检测 FastAPI 服务：单模型一次前向，同时产出文档级与句子级双级结果。

推理优化：CUDA 上 bf16 半精度 + DynamicBatcher 攒批（时间窗口内到达的请求
拼成大 batch 一次 forward，吞吐随并发提升）。
"""

import json
import os
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langdetect import DetectorFactory, detect
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT / "model_entity"), str(_REPO_ROOT / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hier_aidetect_model import HierBertForAiDetect  # noqa: E402,F401
from hier_model_factory import load_hier_model_class  # noqa: E402
from sentence_splitter import split_paragraphs, split_sentences  # noqa: E402

# backbone 路由：默认模型为 xlmr；与 MODEL_DIR 的 checkpoint 类型必须一致，
# 启动时校验（backbone 未加载会直接失败，防止静默随机权重服务）。
MODEL_TYPE = os.environ.get("AI_DETECT_MODEL_TYPE", "xlmr")
MODEL_DIR = os.environ.get(
    "AI_DETECT_MODEL_DIR", str(_REPO_ROOT / "model_outputs_hier" / "run_xlmr" / "best")
)

HOST = os.environ.get("AI_DETECT_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_DETECT_PORT", "8100"))
SERVICE_VERSION = "V2-hier-1"
NEAT_VERSION = "V2h"
MAX_SENTENCES = 256
MAX_BATCH_DOCS = 32

# ============== 推理性能配置 ==============
# CUDA 上用 bf16 半精度（Ampere+ 原生支持，数值比 fp16 稳），显存减半延迟降约一半。
USE_BF16 = True
# 攒批参数：窗口内到达的请求拼批；80G A800 + bf16 显存余量巨大
# （1024 句批峰值 <6GB），上限按 GPU 饱和点（~8 万 token/s）与延迟权衡取值。
BATCH_WINDOW_S = 0.05
BATCH_MAX_DOCS = 16
BATCH_MAX_SENTS = 1024
# 单请求等待攒批结果的兜底超时。
REQUEST_TIMEOUT_S = 60.0

# langdetect 结果稳定化。
DetectorFactory.seed = 0

DOC_CLASS_ENUM = {"human": "HUMAN_ONLY", "ai": "AI_ONLY", "mixed": "MIXED"}
DOC_CLASSES = ["human", "ai", "mixed"]
SENT_CLASSES = ["human", "ai", "paraphrased"]

# (predictedClass, confidenceCategory) → 文案，与上游检测日志的枚举口径一致。
RESULT_MESSAGES = {
    ("human", "high"): "High confidence that the text was written entirely by a human.",
    ("human", "medium"): "Moderate confidence that the text was written entirely by a human.",
    ("ai", "high"): "High confidence that the text was written by AI.",
    ("ai", "medium"): "Moderate confidence that the text was written by AI.",
    ("mixed", "high"): "High confidence that the text mixes human-written and AI-written parts.",
    ("mixed", "medium"): "Moderate confidence that the text mixes human-written and AI-written parts.",
}
_LOW_MESSAGE = (
    "Low confidence: the signal is not strong enough to call this text "
    "human-written or AI-written."
)


def classify_confidence(score: float) -> str:
    """置信度分档阈值：0.33 / 0.6 / 0.8（与上游日志的 confidenceCategory 口径一致）。"""
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.33:
        return "low"
    return "reject"


def _thresholds_raw() -> dict:
    """confidenceThresholdsRaw：分档阈值表（三类结构相同）。"""
    levels = {"reject": 0.33, "low": 0.6, "medium": 0.8}
    return {"identity": {cls: dict(levels) for cls in DOC_CLASSES}}


def _detect_language(text: str) -> str:
    """langdetect 检测语言，异常兜底 en。"""
    try:
        return detect(text)
    except Exception:
        return "en"


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _softmax_2d(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_document(
    text: str,
    paragraphs_meta: list[dict],
    sentences: list[dict],
    sent_origin_probs: np.ndarray,
    doc_probs: np.ndarray,
    language: str,
    document_id: str,
    truncated: bool = False,
) -> dict:
    """组装单个 document 的完整双级结构。

    Args:
        text: 原始输入文本。
        paragraphs_meta: [{"start_sentence_index", "num_sentences"}]。
        sentences: [{"text"}]，与 sent_origin_probs 行对齐。
        sent_origin_probs: (N, 3) [human, ai, paraphrased]。
        doc_probs: (3,) [human, ai, mixed]，已温度校准。
        language: 语言码。
        document_id: 文档 uuid。
        truncated: 句子数是否被截断到 MAX_SENTENCES。

    Returns:
        document dict。
    """
    class_probs = {DOC_CLASSES[i]: float(doc_probs[i]) for i in range(3)}
    pred_idx = int(doc_probs.argmax())
    predicted = DOC_CLASSES[pred_idx]
    confidence = float(doc_probs.max())
    category = classify_confidence(confidence)
    if category in ("low", "reject"):
        message = _LOW_MESSAGE
    else:
        message = RESULT_MESSAGES.get(
            (predicted, category), RESULT_MESSAGES[(predicted, "medium")]
        )

    gen_probs = sent_origin_probs[:, 1] + sent_origin_probs[:, 2]
    sent_objs = []
    for i, s in enumerate(sentences):
        cp = sent_origin_probs[i]
        sent_objs.append({
            "generatedProb": float(gen_probs[i]),
            "sentence": s["text"],
            "perplexity": 0,
            "classProbabilities": {
                SENT_CLASSES[j]: float(cp[j]) for j in range(3)
            },
            "highlightSentenceForAi": bool(gen_probs[i] >= 0.5),
            "specialHighlightType": "polished" if int(cp.argmax()) == 2 else None,
        })

    para_objs = []
    for p in paragraphs_meta:
        seg = gen_probs[
            p["start_sentence_index"]:
            p["start_sentence_index"] + p["num_sentences"]
        ]
        para_objs.append({
            "startSentenceIndex": p["start_sentence_index"],
            "numSentences": p["num_sentences"],
            "completelyGeneratedProb": float(seg.mean()) if len(seg) else 0.0,
        })

    result_message = message + (
        " (Input truncated to 256 sentences.)" if truncated else ""
    )
    return {
        "paragraphs": para_objs,
        "sentences": sent_objs,
        "classProbabilities": class_probs,
        "confidenceThresholdsRaw": _thresholds_raw(),
        "confidenceScoresRaw": {"identity": dict(class_probs)},
        "subclass": {},
        "pageNumber": 0,
        "language": language,
        "inputText": text,
        "documentId": document_id,
        "predictedClass": predicted,
        "confidenceScore": confidence,
        "confidenceCategory": category,
        "documentClassification": DOC_CLASS_ENUM[predicted],
        "resultMessage": result_message,
        "completelyGeneratedProb": float(doc_probs[1]),
        "averageGeneratedProb": float(gen_probs.mean()) if len(gen_probs) else 0.0,
        "overallBurstiness": 0,
        "writingStats": {},
        "version": SERVICE_VERSION,
        "neatVersion": NEAT_VERSION,
    }


def build_response(documents: list[dict]) -> dict:
    """顶层 {version, neatVersion, scanId, documents}。"""
    return {
        "version": SERVICE_VERSION,
        "neatVersion": NEAT_VERSION,
        "scanId": uuid.uuid4().hex,
        "documents": documents,
    }


# 兼容旧名（测试与外部脚本仍按旧名导入）。
build_gptzero_document = build_document
build_gptzero_response = build_response


# ============== 推理 ==============
tokenizer: Optional[AutoTokenizer] = None
model: Optional[object] = None
temperature: float = 1.0
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_LOADED = False
_batcher: Optional["DynamicBatcher"] = None


class DynamicBatcher:
    """请求攒批器：时间窗口内到达的请求拼成大 batch 一次 forward。

    请求线程 submit() 后阻塞等待 event；后台线程单点访问模型（消除多线程
    共享 forward 的竞争），拼批/拆分逻辑与训练 collator 同构。

    Args:
        model: 已加载并 eval 的联合模型。
        tokenizer: 对应 tokenizer。
        device: 计算设备。
        window_s: 攒批时间窗口（秒），首请求最多多等一个窗口。
        max_batch_docs: 单批最大文档数。
        max_batch_sents: 单批最大总句数（显存保护）。
    """

    def __init__(self, model, tokenizer, device: torch.device,
                 window_s: float = BATCH_WINDOW_S,
                 max_batch_docs: int = BATCH_MAX_DOCS,
                 max_batch_sents: int = BATCH_MAX_SENTS) -> None:
        # 防御：batcher 是模型的唯一使用者，确保模型与其输入设备一致，
        # 杜绝"加载后忘 to(device)"导致的 cpu/cuda 混设备运行时错误。
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.window_s = window_s
        self.max_batch_docs = max_batch_docs
        self.max_batch_sents = max_batch_sents
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动后台攒批线程（daemon，随进程退出）。"""
        self._thread = threading.Thread(target=self._loop, daemon=True, name="batcher")
        self._thread.start()

    def submit(self, text: str, sentences: list[dict]) -> dict:
        """提交一篇文档，阻塞等待拼批推理结果。

        Args:
            text: 全文（文档级通道输入）。
            sentences: [{"text"}]（已截断到 MAX_SENTENCES）。

        Returns:
            {"sent_probs": (N,3), "doc_logits": (3,), "truncated": bool}。

        Raises:
            HTTPException: 等待超时（504）或推理异常（500）。
        """
        return self.submit_batch([text], [sentences])[0]

    def submit_batch(self, texts: list[str], sentences_list: list[list[dict]]) -> list[dict]:
        """整批一次性提交（必然拼入同批或按上限自然分批），统一等待结果。

        Args:
            texts: 每篇全文（文档级通道输入）。
            sentences_list: 每篇 [{"text"}]（已截断到 MAX_SENTENCES）。

        Returns:
            与输入同序的结果列表，元素同 submit()。

        Raises:
            HTTPException: 等待超时（504）或推理异常（500）。
        """
        jobs = []
        for text, sentences in zip(texts, sentences_list):
            jobs.append({
                "text": text,
                "sentences": sentences,
                "event": threading.Event(),
                "result": None,
                "error": None,
            })
        for job in jobs:
            self._queue.put(job)
        results = []
        for job in jobs:
            if not job["event"].wait(timeout=REQUEST_TIMEOUT_S):
                raise HTTPException(status_code=504, detail="inference timeout")
            if job["error"] is not None:
                raise HTTPException(
                    status_code=500, detail=f"inference failed: {job['error']}"
                )
            results.append(job["result"])
        return results

    def _loop(self) -> None:
        """后台循环：等首个请求 → 窗口内继续收 → 出批处理。"""
        while True:
            jobs = [self._queue.get()]
            deadline = time.monotonic() + self.window_s
            total_sents = len(jobs[0]["sentences"])
            while (
                len(jobs) < self.max_batch_docs
                and total_sents < self.max_batch_sents
                and time.monotonic() < deadline
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    job = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                jobs.append(job)
                total_sents += len(job["sentences"])
            self.process_jobs(jobs)

    def process_jobs(self, jobs: list[dict]) -> None:
        """拼批一次 forward，结果按篇拆回各 job（公开便于测试）。

        任意异常回填同批所有 job 的 error，保证请求线程不悬挂。
        """
        try:
            sent_texts: list[str] = []
            doc_ids: list[int] = []
            for doc_idx, job in enumerate(jobs):
                for s in job["sentences"]:
                    sent_texts.append(s["text"])
                    doc_ids.append(doc_idx)
            enc = self.tokenizer(
                sent_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            doc_enc = self.tokenizer(
                [job["text"] for job in jobs], padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            batch = {
                "input_ids": enc["input_ids"].to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device),
                "doc_ids": torch.tensor(doc_ids, dtype=torch.long, device=self.device),
                "num_docs": len(jobs),
                "doc_input_ids": doc_enc["input_ids"].to(self.device),
                "doc_attention_mask": doc_enc["attention_mask"].to(self.device),
            }
            with torch.no_grad():
                out = self.model(**batch)
            sent_probs = _softmax_2d(out.logits_origin.float().cpu().numpy())
            doc_logits = out.logits_doc.float().cpu().numpy()
            cursor = 0
            for doc_idx, job in enumerate(jobs):
                n = len(job["sentences"])
                job["result"] = {
                    "sent_probs": sent_probs[cursor : cursor + n],
                    "doc_logits": doc_logits[doc_idx],
                    "truncated": job.get("truncated", False),
                }
                cursor += n
                job["event"].set()
        except Exception as exc:  # noqa: BLE001 回填所有 job，避免请求线程悬挂
            for job in jobs:
                job["error"] = exc
                job["event"].set()


def detect_documents(texts: list[str]) -> list[dict]:
    """批量检测：逐篇分句后整批提交攒批器，组装 documents。

    Args:
        texts: 待检测文本列表（顺序与返回 documents 一致）。

    Returns:
        document dict 列表。
    """
    all_sentences: list[list[dict]] = []
    all_paragraphs: list[list[dict]] = []
    all_truncated: list[bool] = []
    for text in texts:
        paragraphs_meta: list[dict] = []
        sentences: list[dict] = []
        for para in split_paragraphs(text):
            para_sents = split_sentences(para)
            if not para_sents:
                continue
            paragraphs_meta.append({
                "start_sentence_index": len(sentences),
                "num_sentences": len(para_sents),
            })
            sentences.extend({"text": s} for s in para_sents)
        if not sentences:
            raise HTTPException(
                status_code=400, detail="text contains no parseable sentence"
            )
        truncated = len(sentences) > MAX_SENTENCES
        if truncated:
            sentences = sentences[:MAX_SENTENCES]
            paragraphs_meta = [
                p for p in paragraphs_meta if p["start_sentence_index"] < MAX_SENTENCES
            ]
            for p in paragraphs_meta:
                p["num_sentences"] = min(
                    p["num_sentences"], MAX_SENTENCES - p["start_sentence_index"]
                )
        all_sentences.append(sentences)
        all_paragraphs.append(paragraphs_meta)
        all_truncated.append(truncated)

    results = _batcher.submit_batch(texts, all_sentences)
    documents = []
    for text, paragraphs_meta, sentences, truncated, result in zip(
        texts, all_paragraphs, all_sentences, all_truncated, results
    ):
        doc_probs = _softmax_1d(result["doc_logits"] / max(temperature, 1e-6))
        documents.append(build_document(
            text=text,
            paragraphs_meta=paragraphs_meta,
            sentences=sentences,
            sent_origin_probs=result["sent_probs"],
            doc_probs=doc_probs,
            language=_detect_language(text),
            document_id=uuid.uuid4().hex,
            truncated=truncated,
        ))
    return documents


# ============== 统一响应包装 code/data/msg ==============
CODE_SUCCESS = 0
# 业务码 → (HTTP 状态码, 默认 msg)；HTTP 码保留供网关/重试逻辑使用。
BIZ_CODE_BY_STATUS = {
    422: 1001,  # 请求参数校验失败
    400: 1002,  # 文本内容无法处理
    413: 1003,  # 批量数量超限
    503: 1004,  # 模型未就绪
}


def wrap_ok(data: dict) -> dict:
    """成功响应包装。"""
    return {"code": CODE_SUCCESS, "msg": "success", "data": data}


# ============== 接口 ==============
class DetectRequest(BaseModel):
    """检测请求：text 同时接受单字符串与字符串数组（数组即批量，≤32 篇）。"""

    text: "str | list[str]" = Field(..., min_length=1)


class BatchDetectRequest(BaseModel):
    """批量检测请求。"""

    texts: list[str] = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动加载模型（CUDA 上 bf16）并启动攒批线程；关闭随进程释放。"""
    global tokenizer, model, temperature, _MODEL_LOADED, _batcher
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_cls = load_hier_model_class(MODEL_TYPE)
    model, loading_info = model_cls.from_pretrained(
        MODEL_DIR, output_loading_info=True
    )
    # 防呆：missing 键里只允许新分类头（classifier_* / head_*）；backbone 未加载
    # （MODEL_TYPE 与 checkpoint 不匹配）时直接拒绝启动，而不是静默随机权重。
    missing = sorted(loading_info.get("missing_keys", []))
    fatal = [k for k in missing if not k.startswith(("classifier_", "head_"))]
    if fatal:
        raise RuntimeError(
            f"backbone 权重未加载（{len(fatal)} 个缺失键，如 {fatal[:3]}）："
            f"MODEL_TYPE='{MODEL_TYPE}' 与 MODEL_DIR='{MODEL_DIR}' 的 "
            f"checkpoint 类型不匹配，请检查配置"
        )
    model = model.to(device)
    # A800（Ampere+）bf16 半精度：显存减半、单次前向提速约一半；CPU 不支持故仅 CUDA。
    if USE_BF16 and device.type == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()
    t_path = Path(MODEL_DIR) / "calibration.json"
    if t_path.exists():
        temperature = json.loads(t_path.read_text(encoding="utf-8"))["temperature"]
    _batcher = DynamicBatcher(model, tokenizer, device)
    _batcher.start()
    _MODEL_LOADED = True
    dtype_name = next(model.parameters()).dtype
    print(f"使用设备: {device}；dtype={dtype_name}；T={temperature:.4f}；服务就绪")
    yield


app = FastAPI(title="AI 文本检测服务（双级）", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """pydantic 校验失败（422）也包装为 code/data/msg 结构。"""
    return JSONResponse(
        status_code=422,
        content={"code": 1001, "msg": f"请求参数校验失败: {exc.errors()[:2]}", "data": None},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """业务异常包装为 code/data/msg；HTTP 状态码原样保留。"""
    biz_code = BIZ_CODE_BY_STATUS.get(exc.status_code, exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": biz_code, "msg": str(exc.detail), "data": None},
    )


@app.get("/")
def root() -> dict:
    """服务探针（同样走 code/data/msg 统一包装）。"""
    return wrap_ok({"status": "ok", "model_loaded": _MODEL_LOADED})


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    """检测：text 为字符串（单篇）或字符串数组（批量 ≤32 篇，整批一次推理）。"""
    texts = [req.text] if isinstance(req.text, str) else req.text
    if len(texts) > MAX_BATCH_DOCS:
        raise HTTPException(
            status_code=413,
            detail=f"batch size {len(texts)} exceeds limit {MAX_BATCH_DOCS}",
        )
    if not _MODEL_LOADED:
        raise HTTPException(status_code=503, detail="model not loaded")
    if any(not t.strip() for t in texts):
        raise HTTPException(status_code=400, detail="empty text in input")
    documents = detect_documents(texts)
    return wrap_ok(build_response(documents))


@app.post("/detect/batch")
def detect_batch(req: BatchDetectRequest) -> dict:
    """批量检测兼容端点；等价于 /detect 的数组输入。"""
    if not _MODEL_LOADED:
        raise HTTPException(status_code=503, detail="model not loaded")
    if any(not text.strip() for text in req.texts):
        raise HTTPException(status_code=400, detail="empty text in batch")
    documents = detect_documents(req.texts)
    return wrap_ok(build_response(documents))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
