"""split_sentences / split_paragraphs 的多语言行为锁定测试。"""

from sentence_splitter import split_paragraphs, split_sentences


def test_english_basic():
    text = "Hello world. This is fine! Is it? Yes."
    assert split_sentences(text) == [
        "Hello world.",
        "This is fine!",
        "Is it?",
        "Yes.",
    ]


def test_chinese_basic():
    text = "今天天气很好。我们去公园吧！好吗？行。"
    assert split_sentences(text) == ["今天天气很好。", "我们去公园吧！", "好吗？", "行。"]


def test_mixed_language():
    text = "AI is here. 中文也行。Next one!"
    assert split_sentences(text) == ["AI is here.", "中文也行。", "Next one!"]


def test_spanish_inverted_punctuation_and_accents():
    text = "Hola, todo bien. ¿Cómo estás? ¡Qué sorpresa! El niño es pequeño. Ánimo y suerte."
    assert split_sentences(text) == [
        "Hola, todo bien.",
        "¿Cómo estás?",
        "¡Qué sorpresa!",
        "El niño es pequeño.",
        "Ánimo y suerte.",
    ]


def test_portuguese_accented_uppercase():
    text = "São Paulo é grande. Êle gosta de água. Ótimo trabalho! Íamos ao clube."
    assert split_sentences(text) == [
        "São Paulo é grande.",
        "Êle gosta de água.",
        "Ótimo trabalho!",
        "Íamos ao clube.",
    ]


def test_empty_and_blank():
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_no_ending_punct_single_sentence():
    assert split_sentences("no ending punct here") == ["no ending punct here"]


def test_whitespace_normalized():
    text = "First sentence.\n\n\n   Second   sentence\twith spaces."
    assert split_sentences(text) == ["First sentence.", "Second sentence with spaces."]


def test_number_start_after_period():
    text = "Version history. 2023 was the year."
    assert split_sentences(text) == ["Version history.", "2023 was the year."]


def test_split_paragraphs_keeps_inner_text():
    text = "Para one line.\nStill para one.\n\nPara two here."
    assert split_paragraphs(text) == [
        "Para one line.\nStill para one.",
        "Para two here.",
    ]


def test_split_paragraphs_drops_empty():
    assert split_paragraphs("\n\n\n  \nA\n\n\n") == ["A"]
