"""batter_profile.py 的測試：重點是「不偷看未來」與冷啟動處理。

執行: python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_preparation"))

import batter_profile as bp  # noqa: E402
import data_spec as spec     # noqa: E402

A, B, NEWBIE = 1, 2, 3


def pitches(batter, date, n, *, swing=True, whiff=False, bip=False, pulled=None, stand="R",
            family="fastball", zone=5, bat_speed=np.nan, ls=np.nan):
    """產生 n 顆相同的球。pulled=True/False 會放一個對應方向的落點。"""
    hc_x = {True: 60.0, False: 190.0, None: np.nan}[pulled] if stand == "R" else \
           {True: 190.0, False: 60.0, None: np.nan}[pulled]
    return pd.DataFrame({
        "game_date": [date] * n, "batter": batter, "stand": stand, "pitch_family": family,
        "zone": float(zone), "description": "x", "is_swing": swing, "is_whiff": whiff, "is_bip": bip,
        "is_model_target": True, "launch_speed": ls, "hc_x": hc_x, "hc_y": 100.0 if pulled is not None else np.nan,
        "bat_speed": bat_speed, "swing_length": bat_speed / 10, "attack_angle": 10.0,
    })


@pytest.fixture
def interim(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "INTERIM_DIR", tmp_path)
    def write(year, *frames):
        pd.concat(frames, ignore_index=True).to_parquet(tmp_path / f"pitches_{year}.parquet")
    return write


def test_spray_pull_direction():
    df = pd.concat([pitches(A, "2024-05-01", 1, bip=True, pulled=True, stand="R"),
                    pitches(A, "2024-05-01", 1, bip=True, pulled=True, stand="L"),
                    pitches(A, "2024-05-01", 1, bip=True, pulled=False, stand="R")], ignore_index=True)
    pulled, has = bp.spray_pull(df)
    assert list(pulled) == [True, True, False] and has.all()


def test_prev_season_reads_only_previous_year(interim):
    interim(2023, pitches(A, "2023-06-01", 200, whiff=True), pitches(B, "2023-06-01", 200, whiff=False))
    interim(2024, pitches(A, "2024-06-01", 200, whiff=False), pitches(B, "2024-06-01", 200, whiff=True))
    prof = bp.build_prev_season(2024).set_index("batter")
    assert prof.loc[A, "prev_season_whiff_rate"] > prof.loc[B, "prev_season_whiff_rate"]
    assert (prof["source_season"] == 2023).all()


def test_no_bat_tracking_is_nan_not_zero(interim):
    interim(2015, pitches(A, "2015-06-01", 50), pitches(B, "2015-06-01", 50))
    prof = bp.build_prev_season(2016)
    assert prof["prev_season_mean_bat_speed"].isna().all()
    assert not prof["bat_tracking_available"].any()


def test_recent_window_excludes_same_day_and_old_games(interim):
    interim(2024,
            pitches(A, "2024-06-01", 60, bip=True, pulled=False),   # 40 天前：窗口外
            pitches(A, "2024-07-01", 60, bip=True, pulled=True),    # 10 天前：窗口內
            pitches(A, "2024-07-11", 60, bip=True, pulled=False))   # 當天：不可計入
    recent = bp.build_recent(2024, prev=None).set_index("game_date")
    row = recent.loc["2024-07-11"]
    assert row["recent_n_batted"] == 60                  # 只有 7/01 那天
    assert row["recent_window_sufficient"]
    assert row["recent_pull_rate"] > 0.6                 # 7/01 全部拉打，只被少量先驗拉回
    assert recent.loc["2024-06-01", "recent_n_batted"] == 0


def test_recent_falls_back_when_window_too_small(interim):
    interim(2024, pitches(A, "2024-07-01", 10, bip=True, pulled=True),
                  pitches(A, "2024-07-05", 10, bip=True, pulled=True))
    recent = bp.build_recent(2024, prev=None).set_index("game_date")
    row = recent.loc["2024-07-05"]
    assert not row["recent_window_sufficient"]           # 窗口內只有 10 次揮棒
    assert row["recent_pull_rate"] == pytest.approx(row["recent_pull_rate_fastball"])


def test_attach_fills_rookies_with_league(interim, tmp_path):
    interim(2023, pitches(A, "2023-06-01", 300, whiff=True), pitches(B, "2023-06-01", 300, whiff=False))
    bp.build_prev_season(2024).to_parquet(tmp_path / "batter_prev_season_2024.parquet")
    games = pd.DataFrame({"batter": [A, NEWBIE], "game_date": ["2024-04-01", "2024-04-01"]})
    out = bp.attach_batter_profile(games, 2024).set_index("batter")
    league = pd.read_parquet(tmp_path / "batter_prev_season_2024.parquet").set_index("batter").loc[bp.LEAGUE_ROW_ID]
    assert not out.loc[A, "is_rookie"] and out.loc[NEWBIE, "is_rookie"]
    assert out.loc[NEWBIE, "prev_season_whiff_rate"] == pytest.approx(league["prev_season_whiff_rate"])
    assert out.loc[NEWBIE, "prev_season_whiff_rate"] != 0
