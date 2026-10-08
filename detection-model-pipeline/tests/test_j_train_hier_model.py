"""j_train 的 collate 与 2-step 训练冒烟测试（tiny config + 本地合成 tokenizer）。"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_entity"))


def _make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["hello", "world", "ai", "human", "text", "cat", "dog", "runs", "fast", "slow"]
    vocab = {"[PAD]": 0, "[UNK]": 1} | {w: i + 2 for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]"
    )


def _make_samples(n: int = 4):
    return [
        {
            "scan_id": f"s{i}",
            "text": "hello world ai human text",
            "doc_label": [0.9, 0.05, 0.05] if i % 2 == 0 else [0.05, 0.9, 0.05],
            "sentences": [
                {"text": "hello world", "y_origin": [0.9, 0.05, 0.05], "y_human_ai": 0.9},
                {"text": "cat runs fast", "y_origin": [0.1, 0.8, 0.1], "y_human_ai": 0.1},
            ],
        }
        for i in range(n)
    ]


def test_collate_batch_shapes():
    from j_train_hier_model import HierDataset, make_hier_collate_fn

    tokenizer = _make_tokenizer()
    collate = make_hier_collate_fn(tokenizer, max_sent_length=16)
    dataset = HierDataset(_make_samples(2))
    batch = collate([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (4, 3)  # 2 篇 × 2 句
    assert batch["doc_ids"].tolist() == [0, 0, 1, 1]
    assert batch["num_docs"] == 2
    assert batch["labels_doc"].shape == (2, 3)
    assert batch["labels_origin"].shape == (4, 3)
    assert batch["labels_human_ai"].shape == (4, 1)
    # V1 式全文通道：每篇一个 512 截断的全文序列。
    assert batch["doc_input_ids"].shape == (2, 5)
    assert batch["doc_attention_mask"].shape == (2, 5)


def test_oversample_by_language():
    from j_train_hier_model import oversample_by_language

    samples = _make_samples(3)
    samples[0]["gptzero_language"] = "en"
    samples[1]["gptzero_language"] = "es"
    samples[2]["gptzero_language"] = "pt"
    boosted = oversample_by_language(samples, ["es", "pt"], factor=2)
    langs = [s["gptzero_language"] for s in boosted]
    assert langs.count("en") == 1
    assert langs.count("es") == 2 and langs.count("pt") == 2
    # factor=1 或空语言列表时原样返回。
    assert oversample_by_language(samples, ["es"], factor=1) == samples
    assert oversample_by_language(samples, [], factor=3) == samples


def test_oversample_fractional_factor():
    """小数倍：整数部分整份复制 + 小数部分按固定种子抽样补齐。"""
    from j_train_hier_model import oversample_by_language

    samples = _make_samples(4)
    for i, lang in enumerate(["en", "es", "es", "pt"]):
        samples[i]["gptzero_language"] = lang
    boosted = oversample_by_language(samples, ["es", "pt"], factor=1.5)
    langs = [s["gptzero_language"] for s in boosted]
    # 3 篇命中 × 0.5 = 1.5 → 四舍五入抽 2 篇副本，总数 4 + 2 = 6。
    assert len(boosted) == 6
    assert langs.count("en") == 1
    assert langs.count("es") + langs.count("pt") == 5
    # 固定种子下两次调用结果一致（可复现）。
    boosted_again = oversample_by_language(samples, ["es", "pt"], factor=1.5)
    assert [s["scan_id"] for s in boosted] == [s["scan_id"] for s in boosted_again]


def _run_two_step_smoke(model, tokenizer, samples, tmp_path) -> None:
    """通用 2-step 训练冒烟：训练 loss 有限 + evaluate 出双级指标。"""
    from j_train_hier_model import (
        HierDataset,
        HierTrainer,
        compute_hier_metrics,
        make_hier_collate_fn,
    )
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        use_cpu=True,
        # 普通 torch Dataset 必须关闭，否则 RemoveColumnsCollator 按模型签名
        # 删除 sentences/text 等非签名键。
        remove_unused_columns=False,
    )
    trainer = HierTrainer(
        model=model,
        args=args,
        train_dataset=HierDataset(samples),
        eval_dataset=HierDataset(samples),
        data_collator=make_hier_collate_fn(tokenizer, 16),
        compute_metrics=compute_hier_metrics,
    )
    result = trainer.train()
    # transformers 5.x 的 training_loss 是 float 而非 Tensor。
    assert math.isfinite(result.training_loss)
    metrics = trainer.evaluate()
    assert "eval_hier_macro_f1" in metrics


def test_two_step_training_smoke_modernbert(tmp_path):
    from hier_aidetect_model import HierBertForAiDetect
    from transformers import ModernBertConfig

    config = ModernBertConfig(
        vocab_size=20,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    _run_two_step_smoke(
        HierBertForAiDetect(config), _make_tokenizer(), _make_samples(4), tmp_path
    )


def test_two_step_training_smoke_xlmr(tmp_path):
    from hier_xlmr_model import HierXlmrForAiDetect
    from transformers import XLMRobertaConfig

    config = XLMRobertaConfig(
        vocab_size=20,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    _run_two_step_smoke(
        HierXlmrForAiDetect(config), _make_tokenizer(), _make_samples(4), tmp_path
    )


def test_model_factory_routes_backbones():
    from hier_aidetect_model import HierBertForAiDetect
    from hier_model_factory import load_hier_model_class
    from hier_xlmr_model import HierXlmrForAiDetect

    assert load_hier_model_class("modernbert") is HierBertForAiDetect
    assert load_hier_model_class("xlmr") is HierXlmrForAiDetect
    with pytest.raises(ValueError):
        load_hier_model_class("bogus")


def test_from_pretrained_loading_info_roundtrip(tmp_path):
    """覆盖 j_train 的真加载路径：save → from_pretrained(output_loading_info)。

    历史教训：from_pretrained 误传 strict 参数在本地 tiny 直构测试下不可见，
    仅远程真权重训练时暴露（strict 非法，透传 __init__ 后 TypeError）。
    """
    from hier_aidetect_model import HierBertForAiDetect
    from hier_xlmr_model import HierXlmrForAiDetect
    from transformers import ModernBertConfig, XLMRobertaConfig

    mb_config = ModernBertConfig(
        vocab_size=20, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=64, pad_token_id=0,
    )
    xlmr_config = XLMRobertaConfig(
        vocab_size=20, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
    )
    for model_cls, config in (
        (HierBertForAiDetect, mb_config),
        (HierXlmrForAiDetect, xlmr_config),
    ):
        model_cls(config).save_pretrained(str(tmp_path / model_cls.__name__))
        model, info = model_cls.from_pretrained(
            str(tmp_path / model_cls.__name__), output_loading_info=True
        )
        assert isinstance(model, model_cls)
        assert not info["missing_keys"]
        assert not info["unexpected_keys"]


def test_xlmr_loads_from_official_checkpoint_prefix(tmp_path):
    """官方 XLM-R checkpoint 键前缀是 roberta.*，entity 属性名必须与其一致。

    历史教训：entity 属性误命名 self.model 时，from_pretrained 对官方权重
    207 个键全部失配（backbone 随机初始化），本地 save→load 自测无法暴露，
    仅远程加载官方 checkpoint 时显形。
    """
    from hier_xlmr_model import HierXlmrForAiDetect
    from transformers import XLMRobertaConfig, XLMRobertaModel

    config = XLMRobertaConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
    )
    # 官方权重形态：裸 XLMRobertaModel 保存（键前缀 roberta.*，无分类头）。
    backbone = XLMRobertaModel(config)
    backbone.save_pretrained(str(tmp_path / "official_xlmr"))

    model, info = HierXlmrForAiDetect.from_pretrained(
        str(tmp_path / "official_xlmr"), output_loading_info=True
    )
    missing = sorted(info["missing_keys"])
    unexpected = sorted(info["unexpected_keys"])
    # 缺失键必须仅为新分类头；backbone（roberta.*）全部加载成功。
    assert missing, "分类头应为新初始化（missing）"
    assert all(k.startswith("classifier_") for k in missing), missing
    # 权重数值一致，证明非随机初始化。
    assert torch.allclose(
        model.roberta.embeddings.word_embeddings.weight,
        backbone.embeddings.word_embeddings.weight,
    )
    assert torch.allclose(
        model.roberta.encoder.layer[0].attention.self.query.weight,
        backbone.encoder.layer[0].attention.self.query.weight,
    )


def test_early_stop_callback_patience_and_min_epochs():
    """patience 计数、改善归零、min_epochs 拦截三口径。"""
    from types import SimpleNamespace

    from j_train_hier_model import EarlyStopWithMinEpochsCallback

    cb = EarlyStopWithMinEpochsCallback(
        metric_name="eval_hier_macro_f1", patience=2, min_epochs=1,
        greater_is_better=True,
    )
    state = SimpleNamespace(global_step=10, epoch=1.5, best_metric=0.5)
    control = SimpleNamespace(should_training_stop=False)

    # 首评（best_metric 已有历史值 0.5）：0.4 未改善 1/2，未达 min_epochs 只计数。
    cb.on_evaluate(None, state, control, metrics={"eval_hier_macro_f1": 0.4})
    assert not control.should_training_stop and cb.counter == 1
    # 0.6 改善：counter 归零，不停止；模拟 HF 随后把 best 更新为 0.6。
    cb.on_evaluate(None, state, control, metrics={"eval_hier_macro_f1": 0.6})
    assert not control.should_training_stop and cb.counter == 0
    state.best_metric = 0.6
    # 连续两次未改善（2/2）触发停止。
    cb.on_evaluate(None, state, control, metrics={"eval_hier_macro_f1": 0.55})
    cb.on_evaluate(None, state, control, metrics={"eval_hier_macro_f1": 0.52})
    assert control.should_training_stop

    # min_epochs 拦截：epoch 未达标时 patience 耗尽也不停。
    cb2 = EarlyStopWithMinEpochsCallback(
        metric_name="eval_hier_macro_f1", patience=1, min_epochs=2,
        greater_is_better=True,
    )
    state2 = SimpleNamespace(global_step=5, epoch=0.5, best_metric=None)
    control2 = SimpleNamespace(should_training_stop=False)
    cb2.on_evaluate(None, state2, control2, metrics={"eval_hier_macro_f1": 0.9})
    cb2.on_evaluate(None, state2, control2, metrics={"eval_hier_macro_f1": 0.1})
    assert not control2.should_training_stop


def test_step_loss_logger_writes_file(tmp_path):
    from types import SimpleNamespace

    from j_train_hier_model import StepLossLoggerCallback

    log_path = tmp_path / "train_log.txt"
    cb = StepLossLoggerCallback(log_path=str(log_path))
    state = SimpleNamespace(global_step=20)
    cb.on_log(None, state, None, logs={"loss": 1.234})
    cb.on_evaluate(None, state, None, metrics={"eval_loss": 0.5, "eval_hier_macro_f1": 0.7})
    content = log_path.read_text(encoding="utf-8")
    assert "step=20" in content and "train_total=1.2340" in content
    assert "eval_loss=0.5000" in content
