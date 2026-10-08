"""l_eval 的概率/一致率纯函数测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from l_eval_hier_model import (  # noqa: E402
    agreement_rate,
    apply_temperature,
    grouped_agreement,
)


def test_apply_temperature_uniform_when_t_large():
    logits = np.array([[10.0, 0.0, 0.0]])
    probs = apply_temperature(logits, 1e6)
    assert np.allclose(probs, 1 / 3, atol=1e-4)


def test_agreement_rate():
    pred = np.array([0, 1, 2, 1])
    truth = np.array([0, 1, 1, 2])
    assert agreement_rate(pred, truth) == 0.5


def test_grouped_agreement_sorting_and_values():
    pred = np.array([0, 1, 0, 1])
    truth = np.array([0, 1, 1, 1])
    groups = ["en", "en", "zh", "zh"]
    rows = grouped_agreement(pred, truth, groups)
    assert rows == [("en", 2, 1.0), ("zh", 2, 0.5)]
