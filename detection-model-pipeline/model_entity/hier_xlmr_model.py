"""V2 联合多任务检测模型（XLM-R 版）：多语 backbone（en/zh/pt/es 等 100 语）。

结构与损失和 hier_aidetect_model.HierBertForAiDetect 完全一致（句级三头 +
文档级 mean+max 池化头，L = L_doc + α·L_sent），仅 backbone 换为 XLM-RoBERTa。
定向复制损失/池化方法而非抽象公共基类：两个 entity 的 PreTrainedModel 基类不同，
保持独立可演化；改一处时需同步另一处。
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import XLMRobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

from hier_aidetect_model import (  # noqa: F401
    DOC_CLASS_NAMES,
    SENT_ORIGIN_NAMES,
    LABEL_IGNORE,
    HierAiDetectOutput,
)


class HierXlmrForAiDetect(XLMRobertaPreTrainedModel):
    """XLM-R 基座 + 句级三头 + 文档级池化头的联合检测模型（多语冷启动路线）。

    句向量取 attention mean pooling（与 ModernBERT 版 V1 口径一致），不使用
    XLM-R 官方 CLS pooler，保证双级蒸馏语义对齐。
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        # 属性名必须是 roberta：XLM-R 官方 checkpoint 键前缀为 roberta.*，
        # 命名不一致会导致 from_pretrained 整个 backbone 匹配失败（随机初始化）。
        self.roberta = XLMRobertaModel(config)
        # XLMRobertaConfig 的 classifier_dropout 属性存在但默认 None，需回退 0.0。
        self.drop = nn.Dropout(getattr(config, "classifier_dropout", None) or 0.0)
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
        """对 BCE per-element loss 做 LABEL_IGNORE 掩码后求均值（同 ModernBERT 版）。"""
        mask = soft_labels != LABEL_IGNORE
        if not mask.any():
            return None
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
        """基于 attention_mask 的 mean pooling（与 ModernBERT 版严格一致）。"""
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
        """按文档归属做句向量 mean+max 池化（同 ModernBERT 版）。

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
        """前向传播，接口与 HierBertForAiDetect 完全一致。"""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        logits_origin = self.classifier_origin(self.drop(pooled))
        logits_human_ai = self.classifier_human_ai(self.drop(pooled))
        logits_ai_rewrite = self.classifier_ai_rewrite(self.drop(pooled))

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
                full_outputs = self.roberta(
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
