"""fit_temperature 的方向性与恒等性测试（合成 logits）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from k_calibrate_hier_model import fit_temperature  # noqa: E402


def test_overconfident_logits_yield_t_above_one():
    # 正确类 logit=8（过自信），软化应得到 T > 1。
    logits = np.array([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    soft = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    assert fit_temperature(logits, soft) > 1.0


def test_well_calibrated_logits_near_one():
    rng = np.random.default_rng(0)
    # 概率 0.7/0.2/0.1 对应的 logit 直接喂入，T 应接近 1。
    p = np.array([0.7, 0.2, 0.1])
    logits = np.log(p)[None, :]
    soft = np.tile(p, (50, 1)) + rng.normal(0, 0.01, (50, 3))
    soft = np.clip(soft, 1e-6, None)
    soft /= soft.sum(axis=1, keepdims=True)
    t = fit_temperature(np.tile(logits, (50, 1)), soft)
    assert 0.5 < t < 2.0
