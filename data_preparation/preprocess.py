"""
preprocess.py
=============
把 fetch_data.py 抓下來的原始逐球資料，依 data_spec.py 的定義轉成建模用的中間檔。

輸入:  dataset/raw/statcast_{year}.parquet (必要時也讀前一季，只讀少數欄位)
輸出:  dataset/interim/pitches_{year}.parquet

本檔案負責「有棒球意義、不需要從訓練資料估參數」的處理：
  - 標籤 (三段式 hurdle 目標，以及三類擊球結果)
  - 棒球意義的正規化 (好球帶正規化、打者視角的內外角)
  - 情境、打線脈絡、序列特徵
  - 資料品質旗標

刻意「不」做的事 (留給建模階段)：
  - z-score、類別編碼：參數必須只從 train 估
  - 球種族內轉速 z-score (spin_z_within_family)：同上
  - 打者輪廓 (batter_profile)：牽涉時間窗，另由 batter_profile.py 處理
  - 切 train/test：由 splits.py 處理

不刪列、只標記：
  序列特徵依賴「同一打席前面的球」。若在這裡直接刪掉觸擊或缺值的球，
  後面的球算出來的「前一球」就會錯位。所以一律保留，改用旗標，
  由建模階段決定哪些球可以當預測目標。

用法:
    python data_preparation/preprocess.py --season 2024
    python data_preparation/preprocess.py --season 2015-2025
    python data_preparation/preprocess.py --season all
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_spec as spec
from fetch_data import resolve_seasons

PA_KEY = ["game_pk", "at_bat_number"]
SEQ_ORDER = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

# 好球帶半寬 (ft)：17 吋本壘板的一半。data_spec 的 plate_x_norm 定義用這個值。
PLATE_HALF_WIDTH_FT = 17 / 2 / 12
HARD_HIT_MPH = 95.0
N_PREV_PITCHES = 3
N_PREV_PAS = 3

# 缺任一項就無法建立序列 token 的核心欄位
CORE_TRACKING = ["pitch_type", "release_speed", "plate_x", "plate_z", "sz_top", "sz_bot", "pfx_x", "pfx_z"]

# 投手的「速球」基準 (speed_vs_own_fastball 用)
OWN_FASTBALL_FAMILIES = ("fastball", "sinker")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. 讀檔與型別
# ---------------------------------------------------------------------------

def load_season(year: int, columns: list[str] | None = None) -> pd.DataFrame | None:
    path = spec.RAW_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=columns)
    # Savant 回傳 nullable 型別 (Int64/string)，計算前統一轉成 numpy 型別，
    # 避免 pd.NA 在比較運算中傳播成 NA 而非 False。
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or df[col].dtype == object:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float64")
    return df


# ---------------------------------------------------------------------------
# 2. 標籤
# ---------------------------------------------------------------------------

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description"]
    ls = df["launch_speed"]

    df["is_swing"] = desc.isin(spec.SWING_DESCRIPTIONS)
    df["is_whiff"] = desc.isin(spec.WHIFF_DESCRIPTIONS)
    df["is_contact"] = desc.isin(spec.CONTACT_DESCRIPTIONS)
    df["is_take"] = desc.isin(spec.TAKE_DESCRIPTIONS)
    df["is_bip"] = desc == "hit_into_play"
    df["is_bunt"] = desc.isin(spec.BUNT_DESCRIPTIONS)

    # Stage 3：只在量得到初速的觸球上有值，其餘為 NaN (不可填 0)
    df["ev_measured"] = ls.where(desc.isin(spec.EV_MEASURED_DESCRIPTIONS))

    # 三類擊球結果 (與先前研究的 v3/v4 模型相容)：
    #   0 未打進場、1 打進場且初速 <= 95、2 打進場且初速 > 95、-1 打進場但初速缺值 (不可當標籤)
    df["contact3"] = np.select(
        [~df["is_bip"], ls.isna(), ls > HARD_HIT_MPH],
        [0, -1, 2],
        default=1,
    ).astype("int8")

    # 序列 token 用的結果類別 (前幾球的打者反應)
    df["pitch_outcome"] = np.select(
        [desc.isin(["ball", "blocked_ball"]),
         desc == "called_strike",
         df["is_whiff"],
         desc.isin(["foul", "foul_tip", "foul_bunt", "bunt_foul_tip"]),
         df["is_bip"],
         desc == "hit_by_pitch"],
        ["ball", "called_strike", "whiff", "foul", "in_play", "hbp"],
        default="other",
    )
    return df


# ---------------------------------------------------------------------------
# 3. 棒球意義的正規化
# ---------------------------------------------------------------------------

def add_baseball_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["plate_z_norm"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    df["plate_x_norm"] = df["plate_x"] / PLATE_HALF_WIDTH_FT

    # 捕手視角 plate_x 正值 = 一壘側。右打者站三壘側，所以一壘側是他的外角。
    # 翻正後統一為「正 = 外角、負 = 內角」。位移 pfx_x 同樣翻正，保持方向一致。
    side = np.where(df["stand"] == "L", -1.0, 1.0)
    df["plate_x_bv"] = df["plate_x_norm"] * side
    df["pfx_x_bv"] = df["pfx_x"] * side

    # 好球帶判定用 Savant 官方 zone (1-9 = 好球帶內)，不自己用座標猜
    df["in_zone"] = df["zone"].between(1, 9)
    df["pitch_family"] = df["pitch_type"].map(spec.PITCH_FAMILY).fillna("other")
    return df


def add_speed_vs_own_fastball(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """本球球速 - 投手自己的速球均速。

    均速只用「本場之前」的比賽，避免用到當天之後的資訊。
    賽季初還沒有前例時，退回該投手上一季的速球均速；兩者皆無則為 NaN 並標記。
    """
    fb = df[df["pitch_family"].isin(OWN_FASTBALL_FAMILIES) & df["release_speed"].notna()]
    daily = (fb.groupby(["pitcher", "game_date"])["release_speed"]
               .agg(["sum", "count"]).reset_index())

    # 每位投手出賽的每一天 (沒投速球的日子也要有值)，依日期累計「嚴格早於當天」的速球
    games = df[["pitcher", "game_date"]].drop_duplicates()
    games = games.merge(daily, on=["pitcher", "game_date"], how="left").fillna({"sum": 0.0, "count": 0.0})
    games = games.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    grp = games.groupby("pitcher")
    prior_sum = grp["sum"].cumsum() - games["sum"]
    prior_cnt = grp["count"].cumsum() - games["count"]
    games["own_fb_speed"] = prior_sum / prior_cnt.replace(0, np.nan)

    prev = load_season(year - 1, columns=["pitcher", "pitch_type", "release_speed"])
    if prev is not None:
        prev = prev[prev["pitch_type"].map(spec.PITCH_FAMILY).isin(OWN_FASTBALL_FAMILIES)]
        prev_mean = prev.groupby("pitcher")["release_speed"].mean()
        games["own_fb_speed"] = games["own_fb_speed"].fillna(games["pitcher"].map(prev_mean))

    df = df.merge(games[["pitcher", "game_date", "own_fb_speed"]], on=["pitcher", "game_date"], how="left").copy()
    df["own_fb_missing"] = df["own_fb_speed"].isna()
    df["speed_vs_own_fastball"] = df["release_speed"] - df["own_fb_speed"]
    return df


# ---------------------------------------------------------------------------
# 4. 情境與對戰脈絡
# ---------------------------------------------------------------------------

def add_context(df: pd.DataFrame) -> pd.DataFrame:
    df["count_state"] = (df["balls"].astype("Int64").astype(str) + "-"
                         + df["strikes"].astype("Int64").astype(str))

    on1, on2, on3 = (df[c].notna().astype(int) for c in ("on_1b", "on_2b", "on_3b"))
    df["base_state"] = (on1 + 2 * on2 + 4 * on3).astype("int8")   # 0 = 空壘 ... 7 = 滿壘
    df["n_runners"] = (on1 + on2 + on3).astype("int8")
    df["risp"] = (on2 + on3) > 0
    df["base_out_state"] = (df["base_state"] * 3 + df["outs_when_up"]).astype("Int64")

    df = df.merge(assign_lineup_slots(df), on=PA_KEY, how="left")

    # 本場面對這位投手的第幾個打席 (內建欄位不分投手)
    df["nth_pa_vs_pitcher"] = (df.groupby(["game_pk", "batter", "pitcher"])["at_bat_number"]
                                 .rank(method="dense").astype("int8"))

    # 投手本場累計投球數 (含本球)
    df["pitcher_pitch_count"] = df.groupby(["game_pk", "pitcher"]).cumcount() + 1
    return df


def assign_lineup_slots(df: pd.DataFrame) -> pd.DataFrame:
    """推導每個打席的棒次 (1-9)。

    不能單純把打席依序編號再 mod 9：只要有一個打席在資料裡沒有任何一球
    (例如不投球的故意四壞，或 fetch 階段剔除了整個打席)，之後整場的棒次都會錯一格。

    改以「打者身分」為錨點：
      - 已出現過的打者，沿用他第一次出現時的棒次，並把游標重設到這一棒
      - 沒出現過的打者 (先發第一輪或代打)，給游標的下一棒
    這樣缺一個打席只會影響到下一位新打者，遇到已知打者就自動校正回來。
    """
    pa = (df.drop_duplicates(PA_KEY)[["game_pk", "inning_topbot", "at_bat_number", "batter"]]
            .sort_values(["game_pk", "inning_topbot", "at_bat_number"]))
    slots = np.empty(len(pa), dtype=np.int8)
    game = pa["game_pk"].to_numpy()
    half = pa["inning_topbot"].to_numpy()
    batter = pa["batter"].to_numpy()

    seen: dict = {}
    cursor = 0
    for i in range(len(pa)):
        if i == 0 or game[i] != game[i - 1] or half[i] != half[i - 1]:
            seen, cursor = {}, 0
        if batter[i] in seen:
            cursor = seen[batter[i]]
        else:
            cursor = cursor % 9 + 1
            seen[batter[i]] = cursor
        slots[i] = cursor
    return pa[PA_KEY].assign(lineup_slot=slots)


def add_lineup_context(df: pd.DataFrame) -> pd.DataFrame:
    """前幾個打席發生了什麼。單位是打席，只用嚴格更早的打席。"""
    last = df.groupby(PA_KEY, sort=False).tail(1)
    pa = last[["game_pk", "at_bat_number", "inning", "inning_topbot", "events", "launch_speed"]].copy()
    pa = pa.sort_values(["game_pk", "inning_topbot", "at_bat_number"]).reset_index(drop=True)

    ev = pa["events"].fillna("")
    pa["result_class"] = np.select(
        [ev.isin(["strikeout", "strikeout_double_play"]),
         ev.isin(["walk", "intent_walk", "hit_by_pitch", "catcher_interf"]),
         ev == "single",
         ev.isin(["double", "triple", "home_run"])],
        ["k", "bb_hbp", "single", "xbh"],
        default="out",
    )
    pa["reached"] = pa["result_class"].isin(["bb_hbp", "single", "xbh"]) | (ev == "field_error")

    g = pa.groupby(["game_pk", "inning_topbot"], sort=False)
    new_cols = ["inning_batters_faced", "prev_pa_reached_count"]
    for j in range(1, N_PREV_PAS + 1):
        pa[f"prev{j}_pa_events"] = g["events"].shift(j)
        pa[f"prev{j}_pa_result_class"] = g["result_class"].shift(j)
        pa[f"prev{j}_pa_launch_speed"] = g["launch_speed"].shift(j)
        new_cols += [f"prev{j}_pa_events", f"prev{j}_pa_result_class", f"prev{j}_pa_launch_speed"]

    reached_prev = pd.concat([g["reached"].shift(j) for j in range(1, N_PREV_PAS + 1)], axis=1)
    pa["prev_pa_reached_count"] = reached_prev.fillna(False).astype(int).sum(axis=1).astype("int8")

    # 本半局至今已面對的打者數 (不含本打席)
    pa["inning_batters_faced"] = (pa.groupby(["game_pk", "inning", "inning_topbot"]).cumcount()).astype("int8")

    return df.merge(pa[PA_KEY + new_cols], on=PA_KEY, how="left")


# ---------------------------------------------------------------------------
# 5. 序列特徵
# ---------------------------------------------------------------------------

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # 前面各步驟逐欄新增，先整理記憶體布局再大量加欄
    g = df.groupby(PA_KEY, sort=False)

    shift_cols = {
        "pitch_family": "pitch_family",
        "plate_x": "plate_x",
        "plate_z_norm": "plate_z_norm",
        "release_speed": "release_speed",
        "release_spin_rate": "release_spin_rate",
        "pitch_outcome": "description",
        "ev_measured": "launch_speed",
    }
    for k in range(1, N_PREV_PITCHES + 1):
        for src, name in shift_cols.items():
            df[f"prev{k}_{name}"] = g[src].shift(k)

    p1 = {c: g[c].shift(1) for c in ["release_speed", "plate_x_norm", "plate_z_norm",
                                      "pfx_x", "pfx_z", "release_pos_x", "release_pos_z"]}
    df["speed_diff_prev1"] = df["release_speed"] - p1["release_speed"]
    df["plate_dx_prev1"] = df["plate_x_norm"] - p1["plate_x_norm"]
    df["plate_dz_prev1"] = df["plate_z_norm"] - p1["plate_z_norm"]
    df["plate_dist_prev1"] = np.hypot(df["plate_dx_prev1"], df["plate_dz_prev1"])
    df["pfx_dx_prev1"] = df["pfx_x"] - p1["pfx_x"]
    df["pfx_dz_prev1"] = df["pfx_z"] - p1["pfx_z"]
    df["pfx_dist_prev1"] = np.hypot(df["pfx_dx_prev1"], df["pfx_dz_prev1"])
    df["same_family_prev1"] = (df["pitch_family"] == df["prev1_pitch_family"]).where(df["prev1_pitch_family"].notna())
    df["family_pair_prev1"] = (df["prev1_pitch_family"] + ">" + df["pitch_family"]).where(df["prev1_pitch_family"].notna())
    df["release_dist_prev1"] = np.hypot(df["release_pos_x"] - p1["release_pos_x"],
                                        df["release_pos_z"] - p1["release_pos_z"])
    # 出手點幾乎相同時分母趨近 0，設 1 吋 (約 0.083 ft) 下限避免數值爆炸
    df["tunnel_ratio_prev1"] = df["plate_dist_prev1"] / df["release_dist_prev1"].clip(lower=1 / 12)

    # 本打席至今已投的球種族序列 (不含本球)。資料已依打席排序，單次線性掃描即可。
    fam = df["pitch_family"].astype(str).to_numpy()
    new_pa = (g.cumcount() == 0).to_numpy()
    hist = np.empty(len(df), dtype=object)
    cur = ""
    for i in range(len(df)):
        if new_pa[i]:
            cur = ""
        hist[i] = cur
        cur = fam[i] if cur == "" else f"{cur}>{fam[i]}"
    df["pa_pitch_history"] = hist
    df["pitch_index_in_pa"] = g.cumcount()
    return df


# ---------------------------------------------------------------------------
# 6. 資料品質旗標
# ---------------------------------------------------------------------------

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(PA_KEY, sort=False)

    df["missing_core"] = df[CORE_TRACKING].isna().any(axis=1)
    # fetch 階段剔除的球種 (如 KN、EP) 或缺漏會讓 pitch_number 不連續，序列在該處斷裂
    expected = g.cumcount() + 1
    df["pitch_number_gap"] = df["pitch_number"] != expected
    dup = df.duplicated(subset=["game_pk", "at_bat_number", "pitch_number"], keep=False)

    pa_bad = pd.DataFrame({
        "pa_missing_core": g["missing_core"].transform("any"),
        "pa_has_gap": g["pitch_number_gap"].transform("any"),
        "pa_has_bunt": g["is_bunt"].transform("any"),
        "pa_has_dup": dup.groupby([df[k] for k in PA_KEY], sort=False).transform("any"),
    })
    df = pd.concat([df, pa_bad], axis=1)
    df["pa_valid"] = ~pa_bad.any(axis=1)

    # 可當預測目標：打席有效，且本球不是觸球 (觸擊) 或觸身球 (非打者對球的判斷)
    df["is_model_target"] = df["pa_valid"] & ~df["is_bunt"] & (df["description"] != "hit_by_pitch")
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def preprocess_season(year: int) -> Path | None:
    t0 = time.time()
    df = load_season(year)
    if df is None:
        log(f"! 找不到 statcast_{year}.parquet，跳過 (先執行 fetch_data.py --season {year})")
        return None
    log(f"=== {year}: 讀入 {len(df):,} 球 ===")

    df = df.sort_values(SEQ_ORDER, kind="stable").reset_index(drop=True)
    df = add_labels(df)
    df = add_baseball_normalization(df)
    df = add_speed_vs_own_fastball(df, year)
    df = df.sort_values(SEQ_ORDER, kind="stable").reset_index(drop=True)
    df = add_context(df)
    df = add_lineup_context(df)
    df = df.sort_values(SEQ_ORDER, kind="stable").reset_index(drop=True)
    df = add_sequence_features(df)
    df = add_quality_flags(df)

    out = spec.INTERIM_DIR / f"pitches_{year}.parquet"
    spec.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log(f"  -> {out.name}: {len(df):,} 球, {len(df.columns)} 欄, {time.time() - t0:.0f}s")
    report(df)
    return out


def report(df: pd.DataFrame) -> None:
    n = len(df)
    n_pa = df[PA_KEY].drop_duplicates().shape[0]
    tgt = df["is_model_target"]
    log(f"  打席 {n_pa:,} | 有效打席 {df.loc[df['pa_valid'], PA_KEY].drop_duplicates().shape[0]:,} "
        f"| 可當目標的球 {tgt.sum():,} ({tgt.mean():.1%})")
    for flag in ["pa_missing_core", "pa_has_gap", "pa_has_bunt", "pa_has_dup", "own_fb_missing"]:
        log(f"    {flag:<16} {df[flag].mean():6.2%} 的球")
    t = df[tgt]
    log(f"  Stage1 出棒率 {t['is_swing'].mean():.1%} | Stage2 觸球率(揮棒中) "
        f"{t.loc[t['is_swing'], 'is_contact'].mean():.1%} | Stage3 有初速 {t['ev_measured'].notna().sum():,} 球")
    c3 = t["contact3"].value_counts(normalize=True).sort_index()
    log("  三類擊球: " + ", ".join(f"{k}={v:.1%}" for k, v in c3.items())
        + "  (0 未打進場 / 1 弱擊 / 2 強擊 / -1 打進場缺初速)")


def main() -> int:
    ap = argparse.ArgumentParser(description="原始逐球資料 -> 建模用中間檔")
    ap.add_argument("--season", default="all", help="年份 / 區間 / 逗號列舉 / all")
    args = ap.parse_args()

    done = [preprocess_season(int(y)) for y in resolve_seasons(args.season)]
    return 0 if any(done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
