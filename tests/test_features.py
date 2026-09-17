"""features.py 與 splits.py 的測試。

重點：
  - 窗口不跨打席、本球的結果被遮蔽
  - 差異特徵由窗口即時計算：反事實替換一顆球後，下一顆球的速差要跟著變
  - 標準化只用 train 的參數，缺值變 0，旗標不被標準化
  - 切分依時間且不重疊

執行: python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_preparation"))

from modeling import features as F  # noqa: E402
import splits as sp                  # noqa: E402

CFG = F.FeatureConfig()


def make_df(pa_lengths, speeds=None):
    """產生數個打席的最小可用資料。pa_lengths=[3, 2] 表示兩個打席各 3、2 球。"""
    rows = []
    k = 0
    for ab, n in enumerate(pa_lengths, start=1):
        for p in range(1, n + 1):
            speed = speeds[k] if speeds is not None else 90.0 + p
            k += 1
            rows.append({
                "game_date": "2024-05-01", "game_pk": 1, "at_bat_number": ab, "pitch_number": p,
                "pitcher": 9, "batter": 100 + ab, "stand": "R", "p_throws": "R",
                "plate_x_bv": 0.1 * p, "plate_z_norm": 0.5, "release_speed": speed, "effective_speed": speed,
                "speed_vs_own_fastball": speed - 95.0, "pfx_x_bv": 0.3, "pfx_z": 1.0, "release_spin_rate": 2300.0,
                "release_extension": 6.5, "release_pos_x": -2.0, "release_pos_z": 6.0,
                "pitch_family": "fastball", "pitch_outcome": "foul" if p < n else "in_play",
                "description": "foul" if p < n else "hit_into_play", "launch_speed": 88.0 if p < n else 101.0,
                "ev_measured": 88.0 if p < n else 101.0, "is_model_target": True, "use_as_target": True,
                "is_swing": True, "is_contact": True, "contact3": 2 if p == n else 0,
                "count_state": "0-0", "base_out_state": 0, "prev1_pa_result_class": None,
                **{c: 1.0 for c in F.CTX_BASE}, **{c: 0.4 for c in F.PROFILE_PREV + F.PROFILE_RECENT},
                "is_rookie": False, "recent_window_sufficient": True,
            })
    return pd.DataFrame(rows)


def identity_normalizer():
    cols = F.TOKEN_BASE + F.TOKEN_DIFF + F.TOKEN_HIST + CFG.ctx_num
    return F.Normalizer(mean={c: 0.0 for c in cols}, std={c: 1.0 for c in cols}, config=CFG.to_dict())


def test_window_does_not_cross_pa_and_masks_target():
    arr = F.build_arrays(make_df([3, 2]), identity_normalizer(), CFG)
    assert arr["token_num"].shape == (5, CFG.seq_len, len(F.TOKEN_NUM))
    L = CFG.seq_len
    # 序列只含同一打席：第二個打席的第一球只有自己一顆；第一個打席的第 3 球有 3 顆
    assert arr["pad"][3].tolist() == [True] * (L - 1) + [False]
    assert arr["pad"][2].tolist() == [True] * (L - 3) + [False] * 3
    assert arr["pad"][4].tolist() == [True] * (L - 2) + [False] * 2
    assert (arr["token_cat"][:, -1, 1] == F.OUTCOME_VOCAB.index(F.MASK)).all()
    ev = F.TOKEN_NUM.index("ev_measured")
    assert (arr["token_num"][:, -1, ev] == 0).all()        # 本球初速被遮蔽
    assert arr["token_num"][2, -2, ev] == pytest.approx(88.0)  # 歷史球保留


def test_long_pa_keeps_every_pitch():
    n = 12
    arr = F.build_arrays(make_df([n]), identity_normalizer(), CFG)
    assert (~arr["pad"][-1]).sum() == n                     # 第 12 球的序列包含第 1 到第 12 球
    speed = F.TOKEN_NUM.index("release_speed")
    assert arr["token_num"][-1, -n, speed] == pytest.approx(91.0)  # 最前面是本打席第 1 球


def test_matchup_uses_both_hands():
    df = make_df([1, 1, 1, 1])
    df["p_throws"] = ["R", "R", "L", "L"]
    df["stand"] = ["R", "L", "R", "L"]
    arr = F.build_arrays(df, identity_normalizer(), CFG)
    slot = F.CTX_CAT.index("matchup")
    names = [F.MATCHUP_VOCAB[i] for i in arr["ctx_cat"][:, slot]]
    assert names == ["RHP_RHB", "RHP_LHB", "LHP_RHB", "LHP_LHB"]


def test_diffs_follow_counterfactual_replacement():
    base = make_df([3], speeds=[95.0, 85.0, 80.0])
    cf = base.copy()
    cf.loc[1, "release_speed"] = 70.0                      # 反事實：把第二球改成 70 mph
    d = F.TOKEN_NUM.index("d_release_speed")
    a0 = F.build_arrays(base, identity_normalizer(), CFG)
    a1 = F.build_arrays(cf, identity_normalizer(), CFG)
    assert a0["token_num"][2, -1, d] == pytest.approx(-5.0)    # 80 - 85
    assert a1["token_num"][2, -1, d] == pytest.approx(10.0)    # 80 - 70：跟著更新
    assert a1["token_num"][2, -2, d] == pytest.approx(-25.0)   # 70 - 95
    assert a0["token_num"][0, -1, d] == 0.0                    # 打席第一球沒有前一球


def test_normalizer_fills_nan_and_keeps_flags():
    df = make_df([2])
    df.loc[1, "release_spin_rate"] = np.nan
    norm = identity_normalizer()
    norm.mean["release_spin_rate"], norm.std["release_spin_rate"] = 2000.0, 100.0
    arr = F.build_arrays(df, norm, CFG)
    spin = F.TOKEN_NUM.index("release_spin_rate")
    assert arr["token_num"][1, -1, spin] == 0.0                # 缺值 = 訓練平均
    assert arr["token_num"][1, -2, spin] == pytest.approx(3.0) # (2300 - 2000) / 100
    assert arr["token_num"][1, -1, F.TOKEN_NUM.index("has_prev")] == 1.0


def test_label_masks():
    df = make_df([2])
    df.loc[0, ["is_swing", "is_contact"]] = False
    df.loc[0, "ev_measured"] = np.nan
    df.loc[1, "contact3"] = -1
    arr = F.build_arrays(df, identity_normalizer(), CFG)
    assert arr["m_contact"].tolist() == [False, True]
    assert arr["m_ev"].tolist() == [False, True]
    assert arr["m_contact3"].tolist() == [True, False]


def test_event_label_and_mask():
    df = make_df([3])
    df.loc[1, ["description", "launch_speed", "ev_measured"]] = ["swinging_strike", np.nan, np.nan]
    df.loc[2, "launch_speed"] = np.nan                     # 打進場但缺初速
    arr = F.build_arrays(df, identity_normalizer(), CFG)
    assert arr["y_event"][:2].tolist() == [F.EVENTS.index("foul"), F.EVENTS.index("whiff")]
    assert arr["m_event"].tolist() == [True, True, False]


def test_target_mask_limits_samples_but_keeps_context():
    df = make_df([3])
    df["use_as_target"] = [False, False, True]              # 前兩球不當目標，但仍是序列脈絡
    arr = F.build_arrays(df, identity_normalizer(), CFG)
    assert len(arr["ctx_num"]) == 1 and (~arr["pad"][0]).sum() == 3


def test_bat_tracking_excluded_by_default():
    assert not set(F.PROFILE_BAT_TRACKING) & set(F.FeatureConfig().ctx_num)
    assert set(F.PROFILE_BAT_TRACKING) <= set(F.FeatureConfig(use_bat_tracking=True).ctx_num)


def test_splits_must_be_disjoint_and_chronological(monkeypatch):
    sp.check_splits()  # 目前設定必須通過
    monkeypatch.setattr(sp, "SPLITS", {"train": [2016, 2024], "val": [2024], "test": [2025, 2026]})
    with pytest.raises(SystemExit):
        sp.check_splits()
    monkeypatch.setattr(sp, "SPLITS", {"train": [2016, 2023], "val": [2025], "test": [2024, 2026]})
    with pytest.raises(SystemExit):
        sp.check_splits()
