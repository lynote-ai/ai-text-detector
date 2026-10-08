"""联合多任务训练：软标签蒸馏双级概率，L = L_doc + α·L_sent。"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    PrinterCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT / "model_entity"), str(_REPO_ROOT / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hier_aidetect_model import HierBertForAiDetect  # noqa: E402,F401
from hier_model_factory import load_hier_model_class  # noqa: E402

# 冷启动默认权重：按 model-type 路由到公开 HF 仓库（也可传本地快照/自训权重目录）。
DEFAULT_INIT_FROM_BY_MODEL_TYPE = {
    "modernbert": "answerdotai/ModernBERT-base",
    "xlmr": "FacebookAI/xlm-roberta-large",
}
DEFAULT_INIT_FROM = None   # None → 按 --model-type 取上表
DEFAULT_TOKENIZER = None   # None → 与 --init-from 同源
DEFAULT_TRAIN_JSONL = str(_REPO_ROOT / "data" / "train_hier.jsonl")
DEFAULT_VAL_JSONL = str(_REPO_ROOT / "data" / "val_hier.jsonl")
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "model_outputs_hier" / "run_xlmr")

# 各 backbone 的默认学习率：对齐单头基线（2e-5）；xlmr-large 等更大模型可 --lr 1e-5 收紧。
DEFAULT_LR_BY_MODEL_TYPE = {"modernbert": 2e-5, "xlmr": 2e-5}

RANDOM_SEED = 42
DEFAULT_EVAL_STEPS = 50        # ~715 optimizer 步/epoch（oversample 后），约 14 次 eval/epoch
DEFAULT_LOGGING_STEPS = 20
MIN_EPOCHS = 3                 # 冷启动联合任务需充分训练（V1 单头用 2，联合放宽到 3）
EARLY_STOP_PATIENCE = 6        # V1 句子级同款：容忍 origin/ai_rewrite 收敛抖动
WARMUP_RATIO = 0.1             # V1 调优值
CLASSIFIER_DROPOUT = 0.1       # V1 经验：蒸馏样本量有限，分类头需正则
MAX_SENT_LENGTH = 128          # 句子通道（同 V1 句子级）
MAX_DOC_LENGTH = 512           # 全文通道（同 V1 文档级，覆盖 ~85% 文档）


class HierDataset(Dataset):
    """一篇文章一个样本；字段与 i_prepare_hier_dataset 产出一致。"""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class HierCollator:
    """展平文章内句子并生成 doc_ids/num_docs 与双级软标签。"""

    def __init__(self, tokenizer, max_sent_length: int = MAX_SENT_LENGTH,
                 alpha: float = 1.0, max_doc_length: int = MAX_DOC_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.max_sent_length = max_sent_length
        self.alpha = alpha
        self.max_doc_length = max_doc_length

    def __call__(self, features: list[dict]) -> dict:
        sent_texts: list[str] = []
        doc_ids: list[int] = []
        y_origin: list[list[float]] = []
        y_human_ai: list[list[float]] = []
        for doc_idx, f in enumerate(features):
            for s in f["sentences"]:
                sent_texts.append(s["text"])
                doc_ids.append(doc_idx)
                y_origin.append(s["y_origin"])
                y_human_ai.append([s["y_human_ai"]])
        enc = self.tokenizer(
            sent_texts,
            padding=True,
            truncation=True,
            max_length=self.max_sent_length,
            return_tensors="pt",
        )
        # V1 式全文通道：文档头额外获得全文 512 token 编码。
        doc_enc = self.tokenizer(
            [f["text"] for f in features],
            padding=True,
            truncation=True,
            max_length=self.max_doc_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "doc_ids": torch.tensor(doc_ids, dtype=torch.long),
            "num_docs": len(features),
            "doc_input_ids": doc_enc["input_ids"],
            "doc_attention_mask": doc_enc["attention_mask"],
            "labels_doc": torch.tensor(
                [f["doc_label"] for f in features], dtype=torch.float32
            ),
            "labels_origin": torch.tensor(y_origin, dtype=torch.float32),
            "labels_human_ai": torch.tensor(y_human_ai, dtype=torch.float32),
            "alpha": torch.tensor(self.alpha, dtype=torch.float32),
        }


def make_hier_collate_fn(tokenizer, max_sent_length: int = MAX_SENT_LENGTH,
                         alpha: float = 1.0) -> HierCollator:
    """工厂函数，供 Trainer 与测试使用。"""
    return HierCollator(tokenizer, max_sent_length, alpha)


def oversample_by_language(
    samples: list[dict], langs: list[str], factor: float
) -> list[dict]:
    """对指定语言的样本过采样，提升该语言在梯度中的占比。

    支持小数倍：整数部分整份复制，小数部分按固定种子随机抽样补齐（可复现）。
    如 factor=1.5 → 命中样本 50% 额外加一份副本。

    Args:
        samples: 训练样本列表（含 gptzero_language 字段）。
        langs: 需过采样语言码列表（如 ["es", "pt"]）。
        factor: 目标倍数（<=1 或空 langs 时原样返回）。

    Returns:
        追加副本后的新列表，不修改原列表。
    """
    if factor <= 1 or not langs:
        return list(samples)
    lang_set = set(langs)
    boosted = [s for s in samples if s.get("gptzero_language") in lang_set]
    full_copies = int(factor - 1)
    fraction = (factor - 1) - full_copies
    extra = boosted * full_copies
    if fraction > 0:
        rng = random.Random(RANDOM_SEED)
        extra = extra + rng.sample(boosted, int(round(len(boosted) * fraction)))
    print(f"过采样 {sorted(lang_set)}：命中 {len(boosted)} 篇 × "
          f"{full_copies} 整份 + {int(round(len(boosted) * fraction))} 篇抽样副本")
    return list(samples) + extra


def compute_hier_metrics(pred: EvalPrediction) -> dict:
    """双级 macro-F1：文档级与句级 argmax 对软标签 argmax。"""
    from sklearn.metrics import f1_score as _f1

    doc_logits, sent_logits = pred.predictions
    doc_labels, sent_labels = pred.label_ids
    doc_f1 = _f1(
        doc_labels.argmax(axis=1), doc_logits.argmax(axis=1), average="macro"
    )
    sent_f1 = _f1(
        sent_labels.argmax(axis=1), sent_logits.argmax(axis=1), average="macro"
    )
    return {
        "hier_macro_f1": float((doc_f1 + sent_f1) / 2),
        "doc_macro_f1": float(doc_f1),
        "sent_macro_f1": float(sent_f1),
    }


class HierTrainer(Trainer):
    """透传双级 logits/labels 的 prediction_step，驱动 compute_hier_metrics。

    batch 中的 alpha 标量在 compute_loss/prediction_step 中 pop 后传给模型，
    实现 --alpha 消融无需改模型接口。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        alpha = inputs.pop("alpha", None)
        outputs = model(**inputs, alpha=float(alpha) if alpha is not None else 1.0)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        alpha = inputs.pop("alpha", None)
        with torch.no_grad():
            outputs = model(
                **inputs, alpha=float(alpha) if alpha is not None else 1.0
            )
        if prediction_loss_only:
            return (outputs.loss.detach(), None, None)
        return (
            outputs.loss.detach(),
            (outputs.logits_doc.detach(), outputs.logits_origin.detach()),
            (inputs["labels_doc"], inputs["labels_origin"]),
        )


# ============== 训练后评估：双级 classification report + 混淆矩阵（V1 格式） ==============
def evaluate_and_report(model, val_samples: list[dict], collator: HierCollator,
                        batch_size: int, device: torch.device) -> None:
    """在 val 上推理，prettytable 输出文档级与句子级 report + 混淆矩阵。

    软标签→硬标签：真实/预测均取 argmax。

    Args:
        model: 训练完成的联合模型。
        val_samples: 验证集样本列表。
        collator: 与训练同款的批处理 collator。
        batch_size: 评估批大小（按文章数）。
        device: 计算设备。
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    model.eval()
    doc_logits_list: list[np.ndarray] = []
    doc_label_list: list[np.ndarray] = []
    sent_logits_list: list[np.ndarray] = []
    sent_label_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(val_samples), batch_size), desc="最终评估"):
            batch = collator(val_samples[i : i + batch_size])
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            alpha = batch.pop("alpha", None)
            out = model(**batch, alpha=float(alpha) if alpha is not None else 1.0)
            doc_logits_list.append(out.logits_doc.cpu().numpy())
            doc_label_list.append(batch["labels_doc"].cpu().numpy())
            sent_logits_list.append(out.logits_origin.cpu().numpy())
            sent_label_list.append(batch["labels_origin"].cpu().numpy())

    doc_logits = np.concatenate(doc_logits_list)
    doc_labels = np.concatenate(doc_label_list)
    sent_logits = np.concatenate(sent_logits_list)
    sent_labels = np.concatenate(sent_label_list)

    _report_head("文档级三分类 [human/ai/mixed]",
                 doc_labels.argmax(axis=1), doc_logits.argmax(axis=1),
                 ["human", "ai", "mixed"])
    _print_confusion_matrix("文档级混淆矩阵 [行=真实, 列=预测]",
                            doc_labels.argmax(axis=1), doc_logits.argmax(axis=1),
                            ["human", "ai", "mixed"])
    _report_head("句子级三分类 [human/ai/paraphrased]",
                 sent_labels.argmax(axis=1), sent_logits.argmax(axis=1),
                 ["human", "ai", "paraphrased"])
    _print_confusion_matrix("句子级混淆矩阵 [行=真实, 列=预测]",
                            sent_labels.argmax(axis=1), sent_logits.argmax(axis=1),
                            ["human", "ai", "paraphrased"])


def _report_head(title: str, y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: list[str]) -> None:
    """classification_report + accuracy 的 prettytable 排版（V1 移植）。"""
    from sklearn.metrics import accuracy_score, classification_report

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    table = PrettyTable()
    table.title = title
    table.field_names = ["类别", "precision", "recall", "f1-score", "support"]
    table.align["类别"] = "l"
    for col in ("precision", "recall", "f1-score", "support"):
        table.align[col] = "r"
    for name in class_names:
        m = report[name]
        table.add_row(
            [name, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
             f"{m['f1-score']:.4f}", int(m["support"])]
        )
    table.add_row(["---", "---", "---", "---", "---"])
    for agg in ("macro avg", "weighted avg"):
        m = report[agg]
        table.add_row(
            [agg, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
             f"{m['f1-score']:.4f}", int(m["support"])]
        )
    table.add_row(["accuracy", "-", "-", f"{acc:.4f}",
                   int(report["weighted avg"]["support"])])
    print(table)


def _print_confusion_matrix(title: str, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: list[str]) -> None:
    """混淆矩阵 prettytable（行=真实，列=预测，含行列合计，V1 移植）。"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    table = PrettyTable()
    table.title = title
    table.field_names = ["真实\\预测"] + class_names + ["行合计"]
    table.align = "r"
    table.align["真实\\预测"] = "l"
    for i, name in enumerate(class_names):
        row_total = int(cm[i].sum())
        table.add_row(
            [name] + [int(cm[i, j]) for j in range(len(class_names))] + [row_total]
        )
    table.add_row(["---"] * (len(class_names) + 2))
    col_totals = [int(cm[:, j].sum()) for j in range(len(class_names))]
    table.add_row(["列合计"] + col_totals + [int(cm.sum())])
    print(table)


def load_jsonl(path: str) -> list[dict]:
    """流式逐行读取 JSONL。"""
    samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"读 {Path(path).name}", mininterval=2.0):
            if line.strip():
                samples.append(json.loads(line))
    return samples


class EarlyStopWithMinEpochsCallback(TrainerCallback):
    """基于 HF state.best_metric 做 patience 计数，不另存第二套 best 状态（V1 移植）。

    与 HF load_best_model_at_end 共用同一真相源（state.best_metric），避免双套
    best_metric 因浮点比较/eval 时机差异导致早停判断与最终加载的 checkpoint 错位。

    时序说明：on_evaluate 在 HF _determine_best_metric 之前触发，故此处读到的
    state.best_metric 是上一轮的值，正好作为"历史最佳"基准与当前 metrics 比较。

    Args:
        metric_name: 监控的指标名（metrics dict 的 key）。
        patience: 允许连续多少次评估未改善后停止。
        min_epochs: 低于该 epoch 数禁止触发 early stop。
        greater_is_better: True 则越大越优，False 则越小越优。
    """

    def __init__(
        self,
        metric_name: str = "eval_hier_macro_f1",
        patience: int = EARLY_STOP_PATIENCE,
        min_epochs: int = MIN_EPOCHS,
        greater_is_better: bool = True,
    ) -> None:
        self.metric_name = metric_name
        self.patience = patience
        self.min_epochs = min_epochs
        # 与 HF _determine_best_metric 完全一致的算子，消除浮点比较差异。
        self._operator = np.greater if greater_is_better else np.less
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return
        current = metrics[self.metric_name]
        prev_best = state.best_metric
        is_improved = (
            self._operator(current, prev_best) if prev_best is not None else True
        )
        if is_improved:
            self.counter = 0
            print(
                f"[EarlyStop] step={state.global_step} epoch={state.epoch:.2f} "
                f"new best {self.metric_name}={current:.4f} (prev={prev_best})",
                flush=True,
            )
            return
        self.counter += 1
        # epoch 未达 min_epochs 时只计数不停止，保证难任务训练时间。
        if state.epoch < self.min_epochs:
            print(
                f"[EarlyStop] step={state.global_step} epoch={state.epoch:.2f} "
                f"未改善 ({self.counter}/{self.patience})，"
                f"但 epoch < {self.min_epochs} 继续训练",
                flush=True,
            )
            return
        if self.counter >= self.patience:
            control.should_training_stop = True
            print(
                f"[EarlyStop] step={state.global_step} epoch={state.epoch:.2f} "
                f"patience {self.counter}/{self.patience} 耗尽，触发 early stop",
                flush=True,
            )


class StepLossLoggerCallback(TrainerCallback):
    """按 step 打印 train/eval 关键指标行，并追加写入训练日志文件（V1 移植）。

    终端与 tqdm 进度条并存：进度条展示实时进度，本回调提供可回溯的 loss 行。

    Args:
        log_path: 日志文件路径，构造时重置（每次训练覆盖）。
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("", encoding="utf-8")

    def _emit(self, line: str) -> None:
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        parts = [f"step={state.global_step}"]
        if "loss" in logs:
            parts.append(f"train_total={logs['loss']:.4f}")
        self._emit(" | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        parts = [f"step={state.global_step}"]
        for key in (
            "eval_loss",
            "eval_hier_macro_f1",
            "eval_doc_macro_f1",
            "eval_sent_macro_f1",
        ):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        self._emit(" | ".join(parts))


def main() -> None:
    """加载热启动模型与双级数据，按 hier_macro_f1 选优训练。"""
    parser = argparse.ArgumentParser(description="训练联合双级检测模型")
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    parser.add_argument(
        "--model-type", choices=["modernbert", "xlmr"], default="xlmr",
        help="backbone：xlmr=多语（默认，冷启动）；modernbert=英文优先（较快的替代路线）",
    )
    parser.add_argument(
        "--init-from", default=DEFAULT_INIT_FROM,
        help="预训练/热启动权重目录或 HF 仓库名；缺省按 --model-type 取公开 backbone",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="缺省与 --init-from 同源")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--oversample-langs", nargs="*", default=None,
        help="需过采样的语言码（如 es pt），配合 --oversample-factor",
    )
    parser.add_argument("--oversample-factor", type=float, default=2.0,
                        help="目标倍数，支持小数（如 1.5：整份 0 份 + 50% 抽样副本）")
    parser.add_argument("--epochs", type=float, default=10.0,
                        help="epoch 上限，early stop 会提前终止")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None,
                        help="缺省按 model-type 取默认（modernbert 2e-5 / xlmr 2e-5）")
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--dropout", type=float, default=CLASSIFIER_DROPOUT,
                        help="分类头 dropout（V1 经验 0.1，传 0 关闭）")
    parser.add_argument("--doc-class-weights", default="1,1,2",
                        help="文档级 [human,ai,mixed] 损失权重，逗号分隔；"
                             "mixed 加权对抗少数类坍塌")
    parser.add_argument("--sent-class-weights", default="1,1,2",
                        help="句子级 [human,ai,paraphrased] 损失权重，逗号分隔")
    args = parser.parse_args()
    # from_pretrained(None) 会触发误导性的网络解析错误，这里先按 model-type 路由到公开 backbone。
    init_from = args.init_from or DEFAULT_INIT_FROM_BY_MODEL_TYPE[args.model_type]
    tokenizer_path = args.tokenizer or init_from
    lr = args.lr if args.lr is not None else DEFAULT_LR_BY_MODEL_TYPE[args.model_type]
    doc_class_weights = [float(x) for x in args.doc_class_weights.split(",")]
    sent_class_weights = [float(x) for x in args.sent_class_weights.split(",")]

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_cls = load_hier_model_class(args.model_type)
    # 蒸馏样本量有限，分类头 dropout 加强正则。
    config = AutoConfig.from_pretrained(init_from)
    config.classifier_dropout = args.dropout
    # 少数类（mixed / paraphrased）损失权重，随 config 序列化保存供推理端无需重传。
    config.doc_class_weights = doc_class_weights
    config.sent_class_weights = sent_class_weights
    # 不传 strict：HF from_pretrained 无此参数且天然非严格加载
    # （缺失键仅 warning 并随机初始化，恰好覆盖新增分类头场景）。
    model, loading_info = model_cls.from_pretrained(
        init_from, config=config, output_loading_info=True
    )
    # sorted 兜底：部分 transformers 版本的 loading_info 返回 set（不可下标）。
    missing = sorted(loading_info.get("missing_keys", []))
    unexpected = sorted(loading_info.get("unexpected_keys", []))
    print(f"初始化权重 {init_from}: 缺失键 {len(missing)} 个（未在权重中的新分类头）, "
          f"未用键 {len(unexpected)} 个")
    if missing:
        print("  缺失键:", missing[:6], "..." if len(missing) > 6 else "")
    if unexpected:
        print("  未用键:", unexpected[:6], "..." if len(unexpected) > 6 else "")

    train_samples = load_jsonl(args.train_jsonl)
    val_samples = load_jsonl(args.val_jsonl)
    if args.oversample_langs:
        train_samples = oversample_by_language(
            train_samples, args.oversample_langs, args.oversample_factor
        )
    print(f"train {len(train_samples)} 篇 / val {len(val_samples)} 篇 / alpha={args.alpha}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        # eval / save / log 节奏必须协调（V1 经验），否则 early stopping 无法生效。
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # save_strategy="best"：仅在 hier_macro_f1 创新优时存盘，
        # 磁盘上天然只剩持续提升的 checkpoint 序列，无需事后清理。
        save_strategy="best",
        save_total_limit=2,
        metric_for_best_model="hier_macro_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        seed=RANDOM_SEED,
        report_to=[],
        # 显式开启训练/评估进度条（tqdm），与 StepLossLogger 的 loss 行并存。
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    trainer = HierTrainer(
        model=model,
        args=training_args,
        train_dataset=HierDataset(train_samples),
        eval_dataset=HierDataset(val_samples),
        data_collator=make_hier_collate_fn(tokenizer, alpha=args.alpha),
        compute_metrics=compute_hier_metrics,
        callbacks=[
            EarlyStopWithMinEpochsCallback(
                metric_name="eval_hier_macro_f1",
                patience=args.patience,
                min_epochs=args.min_epochs,
                greater_is_better=True,
            ),
            StepLossLoggerCallback(
                log_path=str(Path(args.output_dir) / "train_log.txt")
            ),
        ],
    )
    # 移除默认 PrinterCallback，避免原始 logs dict 与进度条/loss 行重复刷屏。
    trainer.remove_callback(PrinterCallback)

    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    best_dir = Path(args.output_dir) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"最佳模型已保存: {best_dir}")
    print(f"best checkpoint: {trainer.state.best_model_checkpoint}")

    print("在 val 集上输出双级 classification report + 混淆矩阵...")
    evaluate_and_report(
        model, val_samples, make_hier_collate_fn(tokenizer, alpha=args.alpha),
        args.eval_batch_size, device,
    )


if __name__ == "__main__":
    main()
