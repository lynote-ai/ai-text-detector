"""联合模型的 backbone 路由工厂：按 model_type 返回 ModernBERT 或 XLM-R 版 entity 类。"""

from typing import Type

from hier_aidetect_model import HierBertForAiDetect


def load_hier_model_class(model_type: str) -> Type:
    """按 model_type 返回联合模型 entity 类（调用方自行 from_pretrained）。

    Args:
        model_type: "modernbert"（V1 热启动路线）或 "xlmr"（多语冷启动路线）。

    Returns:
        对应的 PreTrainedModel 子类。

    Raises:
        ValueError: model_type 不识别时。
    """
    if model_type == "modernbert":
        return HierBertForAiDetect
    if model_type == "xlmr":
        from hier_xlmr_model import HierXlmrForAiDetect

        return HierXlmrForAiDetect
    raise ValueError(f"未知 model_type: {model_type}，可选 modernbert / xlmr")
