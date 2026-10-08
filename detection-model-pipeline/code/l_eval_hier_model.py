"""联合模型评测：双级分类指标、与上游标签的一致率、可选基线文档模型对照。"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT / "model_entity"), str(_REPO_ROOT / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hier_aidetect_model import (  # noqa: E402,F401
    DOC_CLASS_NAMES,
    SENT_ORIGIN_NAMES,
)
from hier_model_factory import load_hier_model_class  # noqa: E402
from j_train_hier_model import make_hier_collate_fn  # noqa: E402

DEFAULT_MODEL_DIR = str(_REPO_ROOT / "model_outputs_hier" / "run_xlmr" / "best")
DEFAULT_VAL_JSONL = str(_REPO_ROOT / "data" / "val_hier.jsonl")

DOC_CLASS_TO_IDX = {"human": 0, "ai": 1, "mixed": 2}


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T)。"""
    scaled = logits / max(temperature, 1e-6)
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def agreement_rate(pred: np.ndarray, truth: np.ndarray) -> float:
    """逐元素标签一致率。"""
    return float((pred == truth).mean())


def grouped_agreement(
    pred: np.ndarray, truth: np.ndarray, groups: list[str]
) -> list[tuple[str, int, float]]:
    """按组（语言）统计一致率，按组名升序返回 (组, 样本数, 一致率)。"""
    buckets: dict[str, list[list[int]]] = defaultdict(lambda: [[], []])
    for p, t, g in zip(pred, truth, groups):
        buckets[g][0].append(int(p))
        buckets[g][1].append(int(t))
    rows = []
    for g in sorted(buckets):
        ps, ts = np.array(buckets[g][0]), np.array(buckets[g][1])
        rows.append((g, len(ps), agreement_rate(ps, ts)))
    return rows


def _report(title: str, y_true: np.ndarray, y_pred: np.ndarray,
            class_names: list[str]) -> None:
    """classification_report + 混淆矩阵的 prettytable 输出（沿用 V1 风格）。"""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    table = PrettyTable(title=title)
    table.field_names = ["类别", "precision", "recall", "f1-score", "support"]
    table.align["类别"] = "l"
    for name in class_names:
        m = report[name]
        table.add_row(
            [name, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
             f"{m['f1-score']:.4f}", int(m["support"])]
        )
    m = report["macro avg"]
    table.add_row(
        ["macro avg", f"{m['precision']:.4f}", f"{m['recall']:.4f}",
         f"{m['f1-score']:.4f}", int(m["support"])]
    )
    table.add_row(
        ["accuracy", "-", "-", f"{accuracy_score(y_true, y_pred):.4f}",
         int(m["support"])]
    )
    print(table)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_table = PrettyTable(title=f"{title} 混淆矩阵 [行=真实, 列=预测]")
    cm_table.field_names = ["真实\\预测"] + class_names
    cm_table.align["真实\\预测"] = "l"
    for i, name in enumerate(class_names):
        cm_table.add_row([name] + [int(v) for v in cm[i]])
    print(cm_table)


def hier_inference(
    model, tokenizer, samples: list[dict], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """val 批量推理，返回文档 logits 与句级 origin 概率（展平全句）。

    Args:
        model: 已加载联合模型。
        tokenizer: 对应 tokenizer。
        samples: val 联合样本。
        device: 计算设备。

    Returns:
        (doc_logits (N_doc,3), sent_probs (N_sent,3))。
    """
    model.eval()
    collator = make_hier_collate_fn(tokenizer)
    doc_logits_list: list[np.ndarray] = []
    sent_probs_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(samples), 8), desc="val 推理", mininterval=2.0):
        batch = collator(samples[i : i + 8])
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            out = model(**batch)
        doc_logits_list.append(out.logits_doc.cpu().numpy())
        sent_probs_list.append(apply_temperature(out.logits_origin.cpu().numpy(), 1.0))
    return np.concatenate(doc_logits_list), np.concatenate(sent_probs_list)


def main() -> None:
    """评测主流程：指标表 + 一致率表 + 语言分组 + 可选基线对照。"""
    parser = argparse.ArgumentParser(description="评测：双级指标 + 与上游标签一致率")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-type", choices=["modernbert", "xlmr"],
                        default="xlmr")
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    parser.add_argument(
        "--temperature-json", default=None, help="缺省尝试 {model-dir}/calibration.json"
    )
    parser.add_argument(
        "--compare-doc-baseline", default=None,
        help="基线文档模型目录（AutoModelForSequenceClassification 三分类，可选）",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model_cls = load_hier_model_class(args.model_type)
    model = model_cls.from_pretrained(args.model_dir).to(device)

    samples: list[dict] = []
    with open(args.val_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    doc_logits, sent_probs = hier_inference(model, tokenizer, samples, device)
    doc_truth = np.array([np.argmax(s["doc_label"]) for s in samples])
    sent_truth = np.array(
        [np.argmax(sent["y_origin"]) for smp in samples for sent in smp["sentences"]]
    )
    languages = [s["gptzero_language"] or "unknown" for s in samples]
    upstream_doc = np.array(
        [DOC_CLASS_TO_IDX.get(s["gptzero_doc_class"], -1) for s in samples]
    )

    _report("联合模型文档级 [human/ai/mixed]", doc_truth, doc_logits.argmax(1),
            DOC_CLASS_NAMES)
    _report("联合模型句子级 [human/ai/paraphrased]", sent_truth, sent_probs.argmax(1),
            SENT_ORIGIN_NAMES)

    temperature = 1.0
    t_path = Path(args.temperature_json or (Path(args.model_dir) / "calibration.json"))
    if t_path.exists():
        temperature = json.loads(t_path.read_text(encoding="utf-8"))["temperature"]
        print(f"加载温度 T = {temperature:.4f}（{t_path}）")
    doc_probs = apply_temperature(doc_logits, temperature)

    agree_table = PrettyTable(title="与上游标签一致率")
    agree_table.field_names = ["口径", "一致率"]
    agree_table.add_row(
        ["文档级（校准前）", f"{agreement_rate(doc_logits.argmax(1), upstream_doc):.4f}"]
    )
    agree_table.add_row(
        ["文档级（校准后 T）", f"{agreement_rate(doc_probs.argmax(1), upstream_doc):.4f}"]
    )
    agree_table.add_row(
        ["句子级", f"{agreement_rate(sent_probs.argmax(1), sent_truth):.4f}"]
    )
    print(agree_table)

    lang_table = PrettyTable(title="按语言分组文档级一致率（校准后）")
    lang_table.field_names = ["language", "样本数", "一致率"]
    for lang, n, rate in grouped_agreement(doc_probs.argmax(1), upstream_doc, languages):
        lang_table.add_row([lang, n, f"{rate:.4f}"])
    print(lang_table)

    if args.compare_doc_baseline:
        base_tok = AutoTokenizer.from_pretrained(args.compare_doc_baseline)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.compare_doc_baseline
        ).to(device)
        base_model.eval()
        base_preds: list[np.ndarray] = []
        for i in tqdm(range(0, len(samples), 16), desc="基线推理", mininterval=2.0):
            enc = base_tok(
                [s["text"] for s in samples[i : i + 16]],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                base_preds.append(base_model(**enc).logits.cpu().numpy())
        base_pred = np.concatenate(base_preds).argmax(1)
        base_table = PrettyTable(title="文档级一致率基线对照")
        base_table.field_names = ["模型", "一致率"]
        base_table.add_row(
            ["联合模型（校准后）",
             f"{agreement_rate(doc_probs.argmax(1), upstream_doc):.4f}"]
        )
        base_table.add_row(
            ["基线文档模型 argmax", f"{agreement_rate(base_pred, upstream_doc):.4f}"]
        )
        print(base_table)


if __name__ == "__main__":
    main()
