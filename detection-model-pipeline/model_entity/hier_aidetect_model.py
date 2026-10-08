"""V2 联合多任务 AI 文本检测模型：句级三头 + 文档级池化分类头，L = L_doc + α·L_sent。

自包含实现（不 import V1 本地 model_entity / 远程 mmbert_multi_heads），
句级三头对齐 V1 ModernBertForAiDetect 的权重命名，保证 from_pretrained 热启动兼容。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import ModernBertPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertModel,
    ModernBertPredictionHead,
)

# 掩码损失哨兵值：概率恒 >= 0，可安全与真实软标签区分（BCE 无 ignore_index 故手动掩码）。
LABEL_IGNORE: float = -1.0

DOC_CLASS_NAMES = ["human", "ai", "mixed"]
SENT_ORIGIN_NAMES = ["human", "ai", "paraphrased"]


@dataclass
class HierAiDetectOutput(ModelOutput):
    """双级联合检测输出。

    Attributes:
        loss: L_doc + α·L_sent 总损失；无任何标签时为 None。
        logits_origin: (N_sent, 3) 句级三分类 logits，列序 [human, ai, paraphrased]。
        logits_human_ai: (N_sent, 1) 句级二分类 logits，正类 human。
        logits_ai_rewrite: (N_sent, 1) 句级二分类 logits，正类 ai。
        logits_doc: (N_doc, 3) 文档级三分类 logits，列序 [human, ai, mixed]。
        loss_origin/loss_human_ai/loss_ai_rewrite/loss_doc: 分头 loss，仅日志用。
        pooled: (N_sent, H) mean-pooled 句向量。
        hidden_states/attentions: 基座透传。
    """

    loss: Optional[torch.Tensor] = None
    logits_origin: Optional[torch.Tensor] = None
    logits_human_ai: Optional[torch.Tensor] = None
    logits_ai_rewrite: Optional[torch.Tensor] = None
    logits_doc: Optional[torch.Tensor] = None
    loss_origin: Optional[torch.Tensor] = None
    loss_human_ai: Optional[torch.Tensor] = None
    loss_ai_rewrite: Optional[torch.Tensor] = None
    loss_doc: Optional[torch.Tensor] = None
    pooled: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class HierBertForAiDetect(ModernBertPreTrainedModel):
    """ModernBERT 基座 + 句级三头 + 文档级池化头的联合检测模型。

    句级三头与 V1 ModernBertForAiDetect 完全一致（origin 软标签 CE、
    human_ai/ai_rewrite 掩码 BCE）；文档头输入为句向量 mean+max 池化拼接。
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head_origin = ModernBertPredictionHead(config)
        self.head_human_ai = ModernBertPredictionHead(config)
        self.head_ai_rewrite = ModernBertPredictionHead(config)
        # classifier_dropout 在 config 中通常未显式声明，getattr 兜底 0.0（同 V1）。
        self.drop = nn.Dropout(getattr(config, "classifier_dropout", 0.0))
        self.classifier_origin = nn.Linear(config.hidden_size, 3)
        self.classifier_human_ai = nn.Linear(config.hidden_size, 1)
        self.classifier_ai_rewrite = nn.Linear(config.hidden_size, 1)
        # 文档头输入 = [句向量池化 2H ‖ 全文编码 H ‖ 句级概率级联统计 6]。
        self.classifier_doc = nn.Linear(config.hidden_size * 3 + 6, 3)
        # 少数类（mixed / paraphrased）损失权重，None 时等价标准软 CE。
        self.sent_class_weights: Optional[list[float]] = getattr(
            config, "sent_class_weights", None
        )
        self.doc_class_weights: Optional[list[float]] = getattr(
            config, "doc_class_weights", None
        )
        self.post_init()

    def _masked_bce_loss(
        self, logits: torch.Tensor, soft_labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """对 BCE per-element loss 做 LABEL_IGNORE 掩码后求均值。

        Args:
            logits: shape (B, K) 预测 logits。
            soft_labels: shape (B, K) 软标签，无效位为 LABEL_IGNORE。

        Returns:
            标量 loss；整 batch 无有效样本时 None。
        """
        mask = soft_labels != LABEL_IGNORE
        if not mask.any():
            return None
        # 哨兵先置 0 再喂 BCE，避免极端 logits 下 ±inf 与 mask(0) 相乘产生 NaN 污染整 batch。
        safe_labels = torch.where(mask, soft_labels, torch.zeros_like(soft_labels))
        bce = BCEWithLogitsLoss(reduction="none")(logits, safe_labels.float())
        return (bce * mask).sum() / mask.sum().clamp(min=1)

    def _soft_ce_loss(
        self, logits: torch.Tensor, soft_labels: torch.Tensor
    ) -> torch.Tensor:
        """软标签多分类交叉熵（上游软标签概率分布作 target）。"""
        return CrossEntropyLoss()(logits, soft_labels)

    def _origin_ce(
        self, logits: torch.Tensor, soft_labels: torch.Tensor
    ) -> torch.Tensor:
        """origin 三分类入口：应用 sent_class_weights 的加权软 CE。"""
        return self._weighted_soft_ce(logits, soft_labels, self.sent_class_weights)

    def _weighted_soft_ce(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,
        class_weights: Optional[list[float]],
    ) -> torch.Tensor:
        """带类权重的软标签交叉熵；weights 为 None 时等价标准软 CE。

        样本级权重 sw = sum_i(t_i * w_i)（标签与权重的内积）：少数类样本的
        per-sample CE 在加权平均中占比放大。不能按维重加权（w*t*logp / w*t
        会在近 one-hot 标签上分子分母约掉，权重失效）。

        Args:
            logits: (B, C) 分类 logits。
            soft_labels: (B, C) 概率分布。
            class_weights: 长度 C 的权重。

        Returns:
            标量 loss。
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        per_sample_ce = -(soft_labels * log_probs).sum(dim=-1)
        if class_weights is None:
            return per_sample_ce.mean()
        w = torch.as_tensor(
            class_weights, dtype=logits.dtype, device=logits.device
        )
        sample_weight = (soft_labels * w).sum(dim=-1)
        return (sample_weight * per_sample_ce).sum() / sample_weight.sum().clamp(
            min=1e-9
        )

    def _segment_stats(self, sent_probs: torch.Tensor, doc_ids: torch.Tensor,
                       num_docs: int) -> torch.Tensor:
        """按文档聚合句级概率，产出级联统计特征（可微）。

        Args:
            sent_probs: (N_sent, C) 句级概率。
            doc_ids: (N_sent,) 句子所属文档序号。
            num_docs: 文档总数。

        Returns:
            (N_doc, 2C) [各类概率均值 ‖ 各类 argmax 占比]。
        """
        counts = (
            torch.bincount(doc_ids, minlength=num_docs)
            .clamp(min=1)
            .to(sent_probs.dtype)
            .unsqueeze(1)
        )
        prob_sums = sent_probs.new_zeros(num_docs, sent_probs.size(1))
        prob_sums.index_add_(0, doc_ids, sent_probs)
        mean_probs = prob_sums / counts
        one_hot = torch.zeros_like(sent_probs).scatter_(
            1, sent_probs.argmax(dim=1, keepdim=True), 1.0
        )
        oh_sums = one_hot.new_zeros(num_docs, one_hot.size(1))
        oh_sums.index_add_(0, doc_ids, one_hot)
        argmax_frac = oh_sums / counts
        return torch.cat([mean_probs, argmax_frac], dim=1)

    def _mean_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """基于 attention_mask 的 mean pooling（与官方实现严格一致）。"""
        if attention_mask is None:
            attention_mask = torch.ones(
                last_hidden_state.shape[:2],
                device=last_hidden_state.device,
                dtype=torch.bool,
            )
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True).to(last_hidden_state.dtype)
        return summed / counts.clamp(min=1e-9)

    def _segment_pool(
        self, sent_vecs: torch.Tensor, doc_ids: torch.Tensor, num_docs: int
    ) -> torch.Tensor:
        """按文档归属做句向量 mean+max 池化。

        Args:
            sent_vecs: (N_sent, H) 句向量。
            doc_ids: (N_sent,) 每句所属文档序号。
            num_docs: 文档总数。

        Returns:
            (N_doc, 2H)，前半 mean 后半 max 拼接。
        """
        sums = sent_vecs.new_zeros(num_docs, sent_vecs.size(1))
        sums.index_add_(0, doc_ids, sent_vecs)
        counts = (
            torch.bincount(doc_ids, minlength=num_docs)
            .clamp(min=1)
            .to(sent_vecs.dtype)
            .unsqueeze(1)
        )
        mean_vec = sums / counts
        max_vec = sent_vecs.new_full(
            (num_docs, sent_vecs.size(1)), torch.finfo(sent_vecs.dtype).min
        )
        max_vec.scatter_reduce_(
            0,
            doc_ids.unsqueeze(1).expand_as(sent_vecs),
            sent_vecs,
            reduce="amax",
            include_self=True,
        )
        return torch.cat([mean_vec, max_vec], dim=1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        num_docs: Optional[int] = None,
        doc_input_ids: Optional[torch.LongTensor] = None,
        doc_attention_mask: Optional[torch.Tensor] = None,
        labels_doc: Optional[torch.Tensor] = None,
        labels_origin: Optional[torch.Tensor] = None,
        labels_human_ai: Optional[torch.Tensor] = None,
        labels_ai_rewrite: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Tuple | HierAiDetectOutput:
        """前向传播，输出句级三头 + 文档头 logits 与联合损失。

        Args:
            input_ids/attention_mask: 展平句子 batch，shape (N_sent, L)。
            doc_ids: (N_sent,) 句子所属文档序号；None 时跳过文档头。
            num_docs: 文档总数；doc_ids 非 None 时必传。
            doc_input_ids/doc_attention_mask: (N_doc, L_full) 每篇全文 tokenize
                （V1 式全文编码通道）；缺省时零填充退化（仅句池化）。
            labels_doc: (N_doc, 3) 文档软标签 [human, ai, mixed]。
            labels_origin: (N_sent, 3) 句级软标签 [human, ai, paraphrased]。
            labels_human_ai/labels_ai_rewrite: (N_sent, 1) 句级二分类软标签，
                可含 LABEL_IGNORE。
            alpha: 句级损失权重，总损失 L = L_doc + α·L_sent。
            output_hidden_states/output_attentions/return_dict: 同 HF 惯例。

        Returns:
            HierAiDetectOutput（或 tuple）。
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        else:
            pooled = self._mean_pool(last_hidden_state, attention_mask)

        logits_origin = self.classifier_origin(self.drop(self.head_origin(pooled)))
        logits_human_ai = self.classifier_human_ai(self.drop(self.head_human_ai(pooled)))
        logits_ai_rewrite = self.classifier_ai_rewrite(
            self.drop(self.head_ai_rewrite(pooled))
        )

        sent_loss = None
        head_losses: dict[str, Optional[torch.Tensor]] = {
            "loss_origin": None,
            "loss_human_ai": None,
            "loss_ai_rewrite": None,
        }
        # origin 三分类带类权重（少数类 paraphrased 加权），其余二头 BCE 不变。
        for logits, labels, key, loss_fn in (
            (logits_origin, labels_origin, "loss_origin", self._origin_ce),
            (logits_human_ai, labels_human_ai, "loss_human_ai", self._masked_bce_loss),
            (logits_ai_rewrite, labels_ai_rewrite, "loss_ai_rewrite", self._masked_bce_loss),
        ):
            if labels is None:
                continue
            loss = loss_fn(logits, labels)
            if loss is not None:
                sent_loss = loss if sent_loss is None else sent_loss + loss
                head_losses[key] = loss

        logits_doc = None
        loss_doc = None
        if doc_ids is not None:
            sent_pool_feats = self._segment_pool(pooled, doc_ids, num_docs)
            # V1 式全文编码通道：backbone 对全文再前向一次，缺省零填充退化。
            if doc_input_ids is not None:
                full_outputs = self.model(
                    input_ids=doc_input_ids,
                    attention_mask=doc_attention_mask,
                    return_dict=True,
                )
                full_pooled = self._mean_pool(
                    full_outputs.last_hidden_state, doc_attention_mask
                )
            else:
                full_pooled = sent_pool_feats.new_zeros(
                    num_docs, sent_pool_feats.size(1) // 2
                )
            # 级联融合：句级概率统计（可微）提供 mixed 判定的"句子分裂"信号。
            sent_stats = self._segment_stats(
                torch.softmax(logits_origin, dim=-1), doc_ids, num_docs
            )
            doc_feats = torch.cat(
                [sent_pool_feats, full_pooled, sent_stats], dim=1
            )
            logits_doc = self.classifier_doc(self.drop(doc_feats))
            if labels_doc is not None:
                loss_doc = self._weighted_soft_ce(
                    logits_doc, labels_doc, self.doc_class_weights
                )

        total_loss = None
        if sent_loss is not None:
            total_loss = alpha * sent_loss
        if loss_doc is not None:
            total_loss = loss_doc if total_loss is None else total_loss + loss_doc

        if not return_dict:
            return (
                total_loss,
                logits_origin,
                logits_human_ai,
                logits_ai_rewrite,
                logits_doc,
                outputs.hidden_states,
                outputs.attentions,
            )
        return HierAiDetectOutput(
            loss=total_loss,
            logits_origin=logits_origin,
            logits_human_ai=logits_human_ai,
            logits_ai_rewrite=logits_ai_rewrite,
            logits_doc=logits_doc,
            loss_origin=head_losses["loss_origin"],
            loss_human_ai=head_losses["loss_human_ai"],
            loss_ai_rewrite=head_losses["loss_ai_rewrite"],
            loss_doc=loss_doc,
            pooled=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
