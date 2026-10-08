"""多语言分句与分段工具：V2 训练数据构造与推理服务共用，不依赖 nltk。"""

import re

# 英文句末标点要求后随空白 + 新句以大写字母/数字/引号/括号/CJK 字符开头；
# \u00c0-\u00de 是拉丁-1 补充大写区（À-Þ），覆盖 es/pt 的重音大写（Á É Ñ Ã Õ Ç 等），
# ¿¡ 为西语倒标点（句首出现）；中日韩句末标点（含省略号）后无条件切分。
_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\u00c0-\u00de¿¡\"'\(\[\u4e00-\u9fff])"
    r"|(?<=[。！？…])"
)
_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> list[str]:
    """将单段落文本切分为句子列表（空白先归一为单空格）。

    Args:
        text: 任意语言文本，可含换行与多余空白。

    Returns:
        句子字符串列表；空文本返回 []。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    """按空行（可含空白）切分段落，保留段落内部换行。

    Args:
        text: 原始文档文本。

    Returns:
        非空段落字符串列表。
    """
    parts = _PARAGRAPH_PATTERN.split(text)
    return [p for p in (part.strip("\n") for part in parts) if p.strip()]
