"""outcomes.py 的測試：事件樹機率守恆、球數轉移符合棒球規則。

執行: python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling import outcomes as O  # noqa: E402


def probs(**kw):
    base = dict(swing=0.5, called_strike_given_take=0.3, contact_given_swing=0.8,
                in_play_given_contact=0.5, foul_tip_given_foul=0.05, hard_given_in_play=0.4)
    base.update(kw)
    return O.StageProbs(**{k: np.asarray(v, dtype=float) for k, v in base.items()})


def test_event_probs_sum_to_one_and_match_tree():
    ev = O.event_probs(probs(swing=[0.0, 0.47, 1.0]))
    total = sum(ev.values())
    assert np.allclose(total, 1.0)
    assert ev["whiff"][1] == pytest.approx(0.47 * 0.2)
    assert ev["in_play_hard"][1] == pytest.approx(0.47 * 0.8 * 0.5 * 0.4)
    assert ev["ball"][2] == 0 and ev["whiff"][0] == 0


@pytest.mark.parametrize("count, event, expected", [
    ((3, 1), "ball", "walk"),
    ((2, 1), "ball", (3, 1)),
    ((1, 2), "called_strike", "strikeout"),
    ((1, 2), "foul_tip", "strikeout"),     # 兩好球擦棒被捕 = 三振
    ((1, 2), "foul", (1, 2)),              # 兩好球界外：球數不變
    ((1, 1), "foul", (1, 2)),
    ((0, 0), "in_play_hard", "in_play_hard"),
])
def test_transition_rules(count, event, expected):
    assert O.transition(*count, event) == expected


def test_step_conserves_probability():
    ev = {k: float(v) for k, v in O.event_probs(probs()).items()}
    dist = {(0, 0): 1.0}
    ended_total = 0.0
    for _ in range(30):  # 界外可以無限延續，推夠多球讓幾乎所有打席結束
        dist, ended = O.step(dist, {c: ev for c in O.COUNTS})
        ended_total += sum(ended.values())
        assert sum(dist.values()) + ended_total == pytest.approx(1.0)
    assert ended_total > 0.999


def test_event_label_mapping():
    desc = pd.Series(["blocked_ball", "swinging_strike_blocked", "foul_tip", "hit_into_play",
                      "hit_into_play", "foul_bunt", "hit_into_play"])
    ls = pd.Series([np.nan, np.nan, np.nan, 101.0, 80.0, np.nan, np.nan])
    labels = O.event_label(desc, ls)
    assert labels.iloc[:5].tolist() == ["ball", "whiff", "foul_tip", "in_play_hard", "in_play_soft"]
    assert pd.isna(labels.iloc[5])                      # 觸擊不在事件樹內
    assert pd.isna(labels.iloc[6])                      # 打進場缺初速：不可當成弱擊
