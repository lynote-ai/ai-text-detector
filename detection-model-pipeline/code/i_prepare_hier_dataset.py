"""联合蒸馏数据集准备：上游检测服务日志 CSV → 三重去重隔离 → 双级软标签 jsonl。

三重排除防数据泄露：旧训练 scan_id、旧日志 CSV 文本 MD5、新 CSV 内部文本 MD5。
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from prettytable import PrettyTable
from tqdm import tqdm

# 流式读取大 CSV 必须放宽字段长度上限（默认 128K 不够）。
# sys.maxsize 在 Windows 上会溢出 C long，逐步降档到 csv 能接受的最大值。
_max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_field_size)
        break
    except OverflowError:
        _max_field_size //= 10

# 单篇文档参与训练/推理的最大句子数，与服务截断口径一致。
MAX_SENTENCES_PER_DOC = 256
MIN_WORD_COUNT = 10
VAL_RATIO_MOD = 10
VAL_HOLDOUT = 1  # md5(scan_id) % 10 == 1 → val

# 用户自行放置的上游检测日志导出（见 docs/DATA_CONSTRUCTION.md 的 CSV 入口契约）。
DEFAULT_NEW_CSV = "data/origin/detection_log_export.csv"
# 重复运行时可传上一批导出/旧数据集，用于跨批文本去重与 scan_id 泄露排除。
DEFAULT_OLD_CSVS: list[str] = []
DEFAULT_OLD_SCAN_JSONLS: list[str] = []
DEFAULT_OUT_DIR = "data"

# 分布统计的标签列序，与 doc_label / y_origin 的 argmax 对齐。
DOC_CLASSES = ["human", "ai", "mixed"]
SENT_CLASSES = ["human", "ai", "paraphrased"]


def md5_norm_text(text: str) -> str:
    """空白归一后取 MD5，用于跨 scan_id 的重复文本识别。"""
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def iter_csv_rows(path: str) -> Iterator[dict]:
    """流式逐行产出 CSV dict（utf-8-sig 兼容 BOM）。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            yield row


def load_excluded_scan_ids(jsonl_paths: list[str]) -> set[str]:
    """从旧训练 jsonl 收集需排除的 scan_id 集合。"""
    ids: set[str] = set()
    for path in jsonl_paths:
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"读 scan_id {Path(path).name}", mininterval=2.0):
                if line.strip():
                    ids.add(json.loads(line)["scan_id"])
    return ids


def build_text_md5_set(csv_paths: list[str]) -> set[str]:
    """流式扫旧 origin CSV，收集 request_payload 文本的 MD5 集合。"""
    md5_set: set[str] = set()
    for path in csv_paths:
        for row in tqdm(
            iter_csv_rows(path), desc=f"扫旧文本 {Path(path).name}", mininterval=2.0
        ):
            try:
                text = json.loads(row["request_payload"])["document"]
            except (json.JSONDecodeError, KeyError):
                continue
            md5_set.add(md5_norm_text(text))
    return md5_set


def extract_sample(row: dict) -> Optional[dict]:
    """CSV 行 → 联合样本；文本过短、无句子、payload 不合规时返回 None。

    Args:
        row: 含 scan_id/request_payload/response_payload/word_count 的 CSV dict。

    Returns:
        联合样本 dict，或 None。
    """
    try:
        text = json.loads(row["request_payload"])["document"]
        doc = json.loads(row["response_payload"])["documents"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    if int(row.get("word_count") or 0) < MIN_WORD_COUNT:
        return None
    sentences_raw = doc.get("sentences", [])[:MAX_SENTENCES_PER_DOC]
    if not sentences_raw:
        return None
    cp = doc.get("classProbabilities", {})
    doc_label = [cp.get("human", 0.0), cp.get("ai", 0.0), cp.get("mixed", 0.0)]
    sentences = []
    for s in sentences_raw:
        scp = s.get("classProbabilities", {})
        sentences.append({
            "text": s["sentence"],
            "y_origin": [
                scp.get("human", 0.0),
                scp.get("ai", 0.0),
                scp.get("paraphrased", 0.0),
            ],
            "y_human_ai": scp.get("human", 0.0),
        })
    return {
        "scan_id": row["scan_id"],
        "text": text,
        "doc_label": doc_label,
        "sentences": sentences,
        "gptzero_doc_class": doc.get("predictedClass", ""),
        "gptzero_language": doc.get("language", ""),
    }


def is_val(scan_id: str) -> bool:
    """scan_id 哈希确定性划分（固定哈希，可复现）。"""
    digest = hashlib.md5(scan_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % VAL_RATIO_MOD == VAL_HOLDOUT


def new_dist_stats() -> dict:
    """新建数据分布统计容器（流式累加，仅存标量）。"""
    return {
        "doc_class": Counter(),       # doc_label argmax 分布
        "sent_class": Counter(),      # 句子 y_origin argmax 分布
        "lang": Counter(),            # 语言分布
        "word_counts": [],            # 每篇词数
        "sent_counts": [],            # 每篇句子数
        "doc_label_max": [],          # 每篇 doc 软标签 max 概率
        "n_docs": 0,
        "n_sents": 0,
    }


def accumulate_dist_stats(stats: dict, sample: dict) -> None:
    """将一个联合样本的分布信息累加进统计容器。

    Args:
        stats: new_dist_stats() 创建的容器。
        sample: extract_sample 产出的联合样本。
    """
    stats["doc_class"][DOC_CLASSES[int(np_argmax(sample["doc_label"]))]] += 1
    for sent in sample["sentences"]:
        stats["sent_class"][SENT_CLASSES[int(np_argmax(sent["y_origin"]))]] += 1
    stats["lang"][sample["gptzero_language"] or "unknown"] += 1
    stats["word_counts"].append(len(sample["text"].split()))
    stats["sent_counts"].append(len(sample["sentences"]))
    stats["doc_label_max"].append(max(sample["doc_label"]))
    stats["n_docs"] += 1
    stats["n_sents"] += len(sample["sentences"])


def np_argmax(values: list[float]) -> int:
    """不引入 numpy 依赖的 argmax（列表小，内置实现足够）。"""
    return max(range(len(values)), key=values.__getitem__)


def _quantile(sorted_values: list, q: float) -> int:
    """升序列表的分位数（最近邻索引，避免 numpy 依赖）。"""
    if not sorted_values:
        return 0
    return sorted_values[min(int(len(sorted_values) * q), len(sorted_values) - 1)]


def _split_row(name: str, train_n: int, val_n: int, train_key, val_key) -> list:
    """构造 train/val 对比表行：[名称, train数, train占比, val数, val占比]。"""
    train_pct = f"{train_key / max(train_n, 1):.1%}"
    val_pct = f"{val_key / max(val_n, 1):.1%}"
    return [name, train_key, train_pct, val_key, val_pct]


def render_dist_tables(train_stats: dict, val_stats: dict) -> None:
    """打印 train/val 的标签、语言、长度与软标签集中度分布对比表。

    Args:
        train_stats: 训练集统计容器。
        val_stats: 验证集统计容器。
    """
    field_names = ["类别", "train数", "train占比", "val数", "val占比"]

    for title, key, class_names, denom_key in (
        ("文档标签分布 train vs val", "doc_class", DOC_CLASSES, "n_docs"),
        ("句子标签分布 train vs val", "sent_class", SENT_CLASSES, "n_sents"),
    ):
        table = PrettyTable(title=title)
        table.field_names = field_names
        table.align["类别"] = "l"
        for cls in class_names:
            table.add_row(_split_row(
                cls, train_stats[denom_key], val_stats[denom_key],
                train_stats[key].get(cls, 0), val_stats[key].get(cls, 0),
            ))
        print(table)

    lang_table = PrettyTable(title="语言分布 train vs val Top10")
    lang_table.field_names = field_names
    lang_table.align["类别"] = "l"
    top_langs = sorted(
        set(train_stats["lang"]) | set(val_stats["lang"]),
        key=lambda g: -(train_stats["lang"].get(g, 0) + val_stats["lang"].get(g, 0)),
    )[:10]
    for lang in top_langs:
        lang_table.add_row(_split_row(
            lang, train_stats["n_docs"], val_stats["n_docs"],
            train_stats["lang"].get(lang, 0), val_stats["lang"].get(lang, 0),
        ))
    print(lang_table)

    def _length_block(stats: dict) -> list:
        words = sorted(stats["word_counts"])
        sents = sorted(stats["sent_counts"])
        label_max = sorted(stats["doc_label_max"])
        n = max(stats["n_docs"], 1)
        return [
            f"{sum(words) / n:.0f}", f"{_quantile(words, 0.5)}",
            f"{_quantile(words, 0.1)}", f"{_quantile(words, 0.9)}",
            f"{sum(sents) / n:.1f}", f"{_quantile(sents, 0.5)}",
            f"{_quantile(sents, 0.9)}",
            f"{sum(1 for v in label_max if v >= 0.95) / n:.1%}",
            f"{sum(1 for v in label_max if v < 0.6) / n:.1%}",
        ]

    len_table = PrettyTable(title="长度与软标签集中度")
    len_table.field_names = [
        "指标", "train", "val",
    ]
    len_table.align["指标"] = "l"
    metric_names = [
        "词数均值", "词数中位数", "词数p10", "词数p90",
        "句数/篇均值", "句数/篇中位数", "句数/篇p90",
        "doc标签max≥0.95占比", "doc标签max<0.6占比",
    ]
    train_vals = _length_block(train_stats)
    val_vals = _length_block(val_stats)
    for name, tv, vv in zip(metric_names, train_vals, val_vals):
        len_table.add_row([name, tv, vv])
    print(len_table)


def main() -> None:
    """主流程：加载排除基线 → 流式处理新 CSV → 写 train/val jsonl → 统计表。"""
    parser = argparse.ArgumentParser(description="准备 V2 双级蒸馏数据集")
    parser.add_argument("--new-csv", default=DEFAULT_NEW_CSV)
    parser.add_argument("--old-csvs", nargs="*", default=DEFAULT_OLD_CSVS)
    parser.add_argument("--old-scan-jsonls", nargs="*", default=DEFAULT_OLD_SCAN_JSONLS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    excluded_ids = load_excluded_scan_ids(args.old_scan_jsonls)
    seen_md5 = build_text_md5_set(args.old_csvs)
    print(f"排除基线：scan_id {len(excluded_ids)} 个，旧文本 MD5 {len(seen_md5)} 个")

    stats: Counter = Counter()
    train_dist = new_dist_stats()
    val_dist = new_dist_stats()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_f = open(out_dir / "train_hier.jsonl", "w", encoding="utf-8")
    val_f = open(out_dir / "val_hier.jsonl", "w", encoding="utf-8")

    for row in tqdm(iter_csv_rows(args.new_csv), desc="处理新 CSV", mininterval=5.0):
        stats["total"] += 1
        if row["scan_id"] in excluded_ids:
            stats["drop_old_scan_id"] += 1
            continue
        try:
            text = json.loads(row["request_payload"])["document"]
        except (json.JSONDecodeError, KeyError):
            stats["drop_bad_payload"] += 1
            continue
        digest = md5_norm_text(text)
        if digest in seen_md5:
            stats["drop_text_md5"] += 1
            continue
        sample = extract_sample(row)
        if sample is None:
            stats["drop_invalid_sample"] += 1
            continue
        # seen_md5 同集合追加，实现新 CSV 内部去重（旧文本与新内部重复共用一桶）。
        seen_md5.add(digest)
        stats["kept"] += 1
        if is_val(sample["scan_id"]):
            val_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            accumulate_dist_stats(val_dist, sample)
            stats["val"] += 1
        else:
            train_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            accumulate_dist_stats(train_dist, sample)
            stats["train"] += 1
    train_f.close()
    val_f.close()

    table = PrettyTable(title="V2 联合蒸馏数据集统计")
    table.field_names = ["环节", "数量"]
    table.align["环节"] = "l"
    for key in (
        "total",
        "drop_old_scan_id",
        "drop_bad_payload",
        "drop_text_md5",
        "drop_invalid_sample",
        "kept",
        "train",
        "val",
    ):
        table.add_row([key, stats[key]])
    print(table)

    render_dist_tables(train_dist, val_dist)


if __name__ == "__main__":
    main()
