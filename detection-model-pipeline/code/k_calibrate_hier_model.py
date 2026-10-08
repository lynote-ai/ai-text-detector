"""在 val 上为文档三分类头拟合温度缩放系数 T，降低假阳性，参数存 calibration.json。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT / "model_entity"), str(_REPO_ROOT / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hier_aidetect_model import DOC_CLASS_NAMES  # noqa: E402
from hier_model_factory import load_hier_model_class  # noqa: E402
from j_train_hier_model import make_hier_collate_fn  # noqa: E402

DEFAULT_MODEL_DIR = str(_REPO_ROOT / "model_outputs_hier" / "run_xlmr" / "best")
DEFAULT_VAL_JSONL = str(_REPO_ROOT / "data" / "val_hier.jsonl")


def fit_temperature(logits: np.ndarray, soft_labels: np.ndarray) -> float:
    """LBFGS 拟合单参数温度 T，最小化软标签 NLL。

    Args:
        logits: (N, 3) 文档头 logits。
        soft_labels: (N, 3) 上游软标签文档概率。

    Returns:
        最优温度 T（>0）。
    """
    lt = torch.tensor(logits, dtype=torch.float64)
    sl = torch.tensor(soft_labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(lt / log_t.exp(), sl)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def collect_doc_logits(
    model, tokenizer, samples: list[dict], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """在 val 上批量推理，收集文档头 logits 与软标签。

    Args:
        model: 已加载的联合模型。
        tokenizer: 对应 tokenizer。
        samples: val 联合样本列表。
        device: 计算设备。

    Returns:
        (doc_logits (N,3), doc_soft_labels (N,3))。
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    collator = make_hier_collate_fn(tokenizer)
    for i in tqdm(range(0, len(samples), 8), desc="val 推理", mininterval=2.0):
        batch = collator(samples[i : i + 8])
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            out = model(**batch)
        all_logits.append(out.logits_doc.cpu().numpy())
        all_labels.append(batch["labels_doc"].cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def main() -> None:
    """加载模型与 val，拟合温度并写入 {model-dir}/calibration.json。"""
    parser = argparse.ArgumentParser(description="拟合文档头温度缩放")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-type", choices=["modernbert", "xlmr"],
                        default="xlmr")
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    parser.add_argument("--tokenizer", default=None, help="缺省用 --model-dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_dir)
    model_cls = load_hier_model_class(args.model_type)
    model = model_cls.from_pretrained(args.model_dir).to(device)

    samples: list[dict] = []
    with open(args.val_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    logits, labels = collect_doc_logits(model, tokenizer, samples, device)
    temperature = fit_temperature(logits, labels)

    output = {
        "temperature": temperature,
        "doc_class_names": DOC_CLASS_NAMES,
        "fitted_samples": len(samples),
    }
    out_path = Path(args.model_dir) / "calibration.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"T = {temperature:.4f}，已写入 {out_path}")


if __name__ == "__main__":
    main()
