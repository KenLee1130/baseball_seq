"""preprocess.py 的單元測試：用手造的小資料檢查最容易出錯、又最難從結果看出來的地方。

執行: python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_preparation"))

import preprocess as pp  # noqa: E402

NO_PREV_SEASON = 1900  # 不存在的年份：上一季資料讀不到


def pitch(game=1, ab=1, n=1, batter=10, pitcher=90, half="Top", date="2024-04-01", desc="ball",
          ls=np.nan, pt="FF", speed=95.0, x=0.0, z=2.5, stand="R", events=None, inning=1, **kw):
    row = dict(game_pk=game, at_bat_number=ab, pitch_number=n, batter=batter, pitcher=pitcher,
               inning_topbot=half, inning=inning, game_date=date, description=desc, launch_speed=ls,
               pitch_type=pt, release_speed=speed, plate_x=x, plate_z=z, sz_top=3.5, sz_bot=1.5,
               pfx_x=0.5, pfx_z=1.0, release_pos_x=-2.0, release_pos_z=6.0, release_spin_rate=2300.0,
               stand=stand, events=events, zone=5.0, balls=0.0, strikes=0.0, outs_when_up=0.0,
               on_1b=np.nan, on_2b=np.nan, on_3b=np.nan)
    row.update(kw)
    return row


def frame(rows):
    return pd.DataFrame(rows).sort_values(pp.SEQ_ORDER).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 標籤
# ---------------------------------------------------------------------------

def test_labels_follow_spec():
    df = pp.add_labels(frame([
        pitch(n=1, desc="foul", ls=88.0),
        pitch(n=2, desc="foul_tip"),
        pitch(n=3, desc="swinging_strike"),
        pitch(n=4, desc="hit_into_play", ls=101.0),
        pitch(ab=2, n=1, desc="hit_into_play", ls=80.0),
        pitch(ab=3, n=1, desc="hit_into_play", ls=np.nan),
    ]))
    # 界外球算觸球，且有初速時進入 Stage 3
    assert df.loc[0, "is_contact"] and df.loc[0, "ev_measured"] == 88.0
    # foul_tip 是觸球但量不到初速：不可填 0
    assert df.loc[1, "is_contact"] and np.isnan(df.loc[1, "ev_measured"])
    assert df.loc[2, "is_whiff"] and not df.loc[2, "is_contact"]
    # 三類：只有打進場才分強弱；缺初速標為 -1，不默默當成弱擊
    assert list(df["contact3"]) == [0, 0, 0, 2, 1, -1]


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------

def test_batter_view_flips_for_lefties():
    df = pp.add_baseball_normalization(pp.add_labels(frame([
        pitch(ab=1, x=0.5, stand="R"),
        pitch(ab=2, x=0.5, stand="L"),
    ])))
    assert df.loc[0, "plate_x_bv"] > 0 and df.loc[1, "plate_x_bv"] < 0
    assert df.loc[0, "plate_z_norm"] == pytest.approx(0.5)  # (2.5 - 1.5) / (3.5 - 1.5)


def test_own_fastball_speed_uses_only_earlier_games():
    df = frame([
        pitch(game=1, date="2024-04-01", speed=90.0),
        pitch(game=2, date="2024-04-05", speed=96.0),
        pitch(game=2, ab=2, date="2024-04-05", speed=85.0, pt="CH"),
    ])
    df = pp.add_speed_vs_own_fastball(pp.add_baseball_normalization(pp.add_labels(df)), NO_PREV_SEASON)
    df = df.sort_values(pp.SEQ_ORDER).reset_index(drop=True)
    assert df.loc[0, "own_fb_missing"]                        # 第一場沒有前例，也沒有上一季
    assert df.loc[1, "own_fb_speed"] == pytest.approx(90.0)   # 不含當天 96 mph
    assert df.loc[2, "speed_vs_own_fastball"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# 棒次與打線脈絡
# ---------------------------------------------------------------------------

def test_lineup_slot_recovers_from_missing_pa():
    # 打者 1..9 第一輪，第二輪 3 號打者的打席在資料裡整個缺失 (例如故意四壞)
    rows = [pitch(ab=i, batter=100 + i) for i in range(1, 10)]
    rows += [pitch(ab=ab, batter=100 + b) for ab, b in zip((10, 11, 12, 13), (1, 2, 4, 5))]
    df = pp.add_context(frame(rows))
    slots = df.drop_duplicates(pp.PA_KEY).set_index("at_bat_number")["lineup_slot"]
    assert list(slots.loc[[10, 11, 12, 13]]) == [1, 2, 4, 5]   # mod 9 的舊作法會給 1, 2, 3, 4


def test_lineup_context_never_sees_current_pa():
    rows = [
        pitch(ab=1, n=1, desc="hit_into_play", ls=105.0, events="home_run"),
        pitch(ab=2, n=1, desc="called_strike"),
        pitch(ab=2, n=2, desc="swinging_strike", events="strikeout"),
        pitch(ab=3, n=1, desc="ball"),
    ]
    df = pp.add_lineup_context(pp.add_context(pp.add_labels(frame(rows))))
    df = df.sort_values(pp.SEQ_ORDER).reset_index(drop=True)
    assert pd.isna(df.loc[0, "prev1_pa_events"])
    assert df.loc[1, "prev1_pa_events"] == "home_run" and df.loc[2, "prev1_pa_events"] == "home_run"
    assert df.loc[3, "prev1_pa_events"] == "strikeout" and df.loc[3, "prev2_pa_result_class"] == "xbh"
    assert df.loc[3, "prev_pa_reached_count"] == 1


# ---------------------------------------------------------------------------
# 序列
# ---------------------------------------------------------------------------

def build_sequence(rows):
    df = pp.add_baseball_normalization(pp.add_labels(frame(rows)))
    return pp.add_sequence_features(df).sort_values(pp.SEQ_ORDER).reset_index(drop=True)


def test_sequence_shift_stays_within_pa():
    df = build_sequence([
        pitch(ab=1, n=1, speed=95.0, pt="FF"),
        pitch(ab=1, n=2, speed=85.0, pt="SL"),
        pitch(ab=2, n=1, speed=90.0, pt="CH"),
    ])
    assert df.loc[1, "speed_diff_prev1"] == pytest.approx(-10.0)
    assert df.loc[1, "family_pair_prev1"] == "fastball>slider"
    # 新打席的第一球不可以看到上一個打席的球
    assert pd.isna(df.loc[2, "prev1_release_speed"]) and pd.isna(df.loc[2, "speed_diff_prev1"])
    assert list(df["pa_pitch_history"]) == ["", "fastball", ""]


def test_gap_flags_whole_pa():
    df = pp.add_quality_flags(build_sequence([
        pitch(ab=1, n=1), pitch(ab=1, n=3),   # 第 2 球在 fetch 階段被剔除
        pitch(ab=2, n=1), pitch(ab=2, n=2, desc="foul_bunt"),
    ]))
    assert not df.loc[df["at_bat_number"] == 1, "pa_valid"].any()
    assert not df.loc[df["at_bat_number"] == 2, "pa_valid"].any()
    assert not df["is_model_target"].any()
