"""
batter_profile.py
=================
依 data_spec.DERIVED_FEATURES["batter_profile"] 建立打者輪廓。兩種時間基準、兩份輸出：

  1. 上一季輪廓 (prev_season_*)
     輸入:  dataset/interim/pitches_{Y-1}.parquet
     輸出:  dataset/interim/batter_prev_season_{Y}.parquet   (一位打者一列，供賽季 Y 使用)
     內容:  揮棒機制、各球種族 x 區域的揮棒/揮空率、各球種族追打率、擊球品質、拉打率、
            好球帶上/中/下的強擊率

  2. 近期輪廓 (recent_*)
     輸入:  dataset/interim/pitches_{Y}.parquet (+ 上一季輪廓作為退路)
     輸出:  dataset/interim/batter_recent_{Y}.parquet        (一位打者一天一列)
     內容:  「本場比賽日之前」30 天內的拉打率 (整體與各球種族)

防止資料洩漏的規則 (tests/test_batter_profile.py 會檢查)：
  - 上一季輪廓只讀 Y-1 的檔案。
  - 近期輪廓的窗口是 [比賽日 - 30 天, 比賽日)，不含比賽日當天 (雙重賽的第一場也不算)。

收縮 (shrinkage)：所有比率都用 (k + w * 先驗) / (n + w) 往聯盟平均拉，樣本越少越接近聯盟。
冷啟動：上一季沒出賽的打者不在表內；合併時填聯盟平均並設 is_rookie = True，絕不填 0。

用法:
    python data_preparation/batter_profile.py --season 2025
    python data_preparation/batter_profile.py --season all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_spec as spec
from fetch_data import resolve_seasons

LEAGUE_ROW_ID = -1                     # 聯盟平均列的 batter 代號，冷啟動時使用
FAMILIES = ("fastball", "sinker", "cutter", "slider", "curveball", "changeup")
FAST_SWING_MPH = 75.0                  # Statcast 官方 fast-swing 門檻
HARD_HIT_MPH = 95.0

# Savant zone 收斂成 4 個區域。13 格 x 6 球種族 x 2 指標會讓每格樣本太薄。
ZONE_REGION = {1: "high", 2: "high", 3: "high",
               4: "mid", 5: "mid", 6: "mid",
               7: "low", 8: "low", 9: "low",
               11: "chase", 12: "chase", 13: "chase", 14: "chase"}
REGIONS = ("high", "mid", "low", "chase")

# 收縮權重：相當於「多少個樣本的聯盟先驗」
W_PITCHES = 100.0     # 以投球數為分母的比率 (揮棒率、追打率)
W_SWINGS = 50.0       # 以揮棒數為分母 (揮空率、快速揮棒率)
W_BIP = 40.0          # 以擊球數為分母 (強擊率、拉打率、初速)

# 拉打率噴射角 (data_spec: recent_pull_rate)
SPRAY_ORIGIN_X, SPRAY_ORIGIN_Y = 125.42, 198.27
PULL_ANGLE_DEG = 15.0
RECENT_WINDOW = "30D"
RECENT_MIN_SWINGS = 50
W_RECENT = 20.0       # 近期拉打率往上一季值收縮的權重

PROFILE_COLUMNS = [
    "game_date", "batter", "stand", "pitch_family", "zone", "description",
    "is_swing", "is_whiff", "is_bip", "is_model_target", "launch_speed",
    "hc_x", "hc_y", "bat_speed", "swing_length", "attack_angle",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_interim(year: int) -> pd.DataFrame | None:
    path = spec.INTERIM_DIR / f"pitches_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=PROFILE_COLUMNS)
    # 只用可當建模目標的球：觸擊、無效打席不代表打者一般的反應
    df = df[df["is_model_target"]].copy()
    df["region"] = df["zone"].map(ZONE_REGION)
    df["is_pulled"], df["has_spray"] = spray_pull(df)
    df["is_hard"] = df["is_bip"] & (df["launch_speed"] > HARD_HIT_MPH)
    return df


def spray_pull(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """回傳 (是否拉打, 是否有落點)。只有打進場且有落點座標的球才有定義。"""
    angle = np.degrees(np.arctan2(df["hc_x"] - SPRAY_ORIGIN_X, SPRAY_ORIGIN_Y - df["hc_y"]))
    has = df["is_bip"] & df["hc_x"].notna() & df["hc_y"].notna()
    # 右打者拉打是左外野方向 (噴射角為負)；左打者相反
    pulled = np.where(df["stand"] == "L", angle > PULL_ANGLE_DEG, angle < -PULL_ANGLE_DEG)
    return pd.Series(pulled & has, index=df.index), has


def shrink(k, n, prior, w):
    return (k + w * prior) / (n + w)


# ---------------------------------------------------------------------------
# 1. 上一季輪廓
# ---------------------------------------------------------------------------

def _rate_block(df: pd.DataFrame, by: list[str], num: str, den: str | None, prior: float, w: float,
                name: str) -> pd.Series:
    """計算一個收縮比率。den=None 表示分母是球數。"""
    g = df.groupby(by)
    k = g[num].sum()
    n = g.size() if den is None else g[den].sum()
    return shrink(k, n, prior, w).rename(name)


def build_prev_season(season: int) -> pd.DataFrame | None:
    """用 season - 1 的資料，建立賽季 season 要用的打者輪廓。"""
    src_year = season - 1
    df = load_interim(src_year)
    if df is None:
        return None

    batters = pd.Index(sorted(df["batter"].unique()), name="batter")
    out = pd.DataFrame(index=batters)
    swings, bip, chase = df[df["is_swing"]], df[df["is_bip"]], df[df["region"] == "chase"]
    spray = df[df["has_spray"]]

    league = {
        "swing": df["is_swing"].mean(),
        "whiff": swings["is_whiff"].mean(),
        "chase": chase["is_swing"].mean(),
        "bip": df["is_bip"].mean(),
        "hard": bip["is_hard"].mean(),
        "pull": spray["is_pulled"].mean(),
        "ev": bip["launch_speed"].mean(),
    }

    out["prev_season_n_pitches"] = df.groupby("batter").size()
    out["prev_season_n_swings"] = swings.groupby("batter").size()
    out["prev_season_swing_rate"] = _rate_block(df, ["batter"], "is_swing", None, league["swing"], W_PITCHES, "x")
    out["prev_season_whiff_rate"] = _rate_block(swings, ["batter"], "is_whiff", None, league["whiff"], W_SWINGS, "x")
    out["prev_season_chase_rate"] = _rate_block(chase, ["batter"], "is_swing", None, league["chase"], W_PITCHES, "x")
    out["prev_season_bip_rate"] = _rate_block(df, ["batter"], "is_bip", None, league["bip"], W_PITCHES, "x")
    out["prev_season_hardhit_rate"] = _rate_block(bip, ["batter"], "is_hard", None, league["hard"], W_BIP, "x")
    out["prev_season_pull_rate"] = _rate_block(spray, ["batter"], "is_pulled", None, league["pull"], W_BIP, "x")

    # 平均初速：只用打進場的球。界外球的初速在 2015 年只有約兩成量得到，
    # 納入會讓不同年份的輪廓不可比。
    ev = bip.dropna(subset=["launch_speed"]).groupby("batter")["launch_speed"].agg(["sum", "count"])
    out["prev_season_ev"] = shrink(ev["sum"], ev["count"], league["ev"], W_BIP)

    # -- 揮棒機制：只有該年份有 bat tracking 時才計算，否則為 NaN (不是 0) --
    has_bat_tracking = src_year >= spec.COLUMN_AVAILABILITY["bat_speed"]
    out["bat_tracking_available"] = has_bat_tracking
    if has_bat_tracking:
        tracked = swings.dropna(subset=["bat_speed"])
        lg_fast = (tracked["bat_speed"] >= FAST_SWING_MPH).mean()
        g = tracked.groupby("batter")
        n = g.size()
        for col, name in [("bat_speed", "prev_season_mean_bat_speed"),
                          ("swing_length", "prev_season_mean_swing_length"),
                          ("attack_angle", "prev_season_mean_attack_angle")]:
            out[name] = shrink(g[col].sum(), n, tracked[col].mean(), W_SWINGS)
        out["prev_season_fast_swing_rate"] = shrink(
            g["bat_speed"].apply(lambda s: (s >= FAST_SWING_MPH).sum()), n, lg_fast, W_SWINGS)
        league.update({"bat_speed": tracked["bat_speed"].mean(), "swing_length": tracked["swing_length"].mean(),
                       "attack_angle": tracked["attack_angle"].mean(), "fast": lg_fast})
    else:
        for name in ["prev_season_mean_bat_speed", "prev_season_mean_swing_length",
                     "prev_season_mean_attack_angle", "prev_season_fast_swing_rate"]:
            out[name] = np.nan

    # -- 好球帶上 / 中 / 下的強擊率 (打進場的球)。不分球種族：18 格會讓每格擊球樣本太少 --
    for reg in ("high", "mid", "low"):
        cell = bip[bip["region"] == reg]
        out[f"prev_season_hardhit_rate_{reg}"] = _rate_block(
            cell, ["batter"], "is_hard", None, cell["is_hard"].mean(), W_BIP / 2, "x")

    # -- 各球種族：追打率、強擊率、平均初速 --
    for fam in FAMILIES:
        f_all, f_bip = df[df["pitch_family"] == fam], bip[bip["pitch_family"] == fam]
        f_chase, f_swing = chase[chase["pitch_family"] == fam], swings[swings["pitch_family"] == fam]
        out[f"prev_season_chase_rate_{fam}"] = _rate_block(
            f_chase, ["batter"], "is_swing", None, f_chase["is_swing"].mean(), W_PITCHES / 2, "x")
        out[f"prev_season_whiff_rate_{fam}"] = _rate_block(
            f_swing, ["batter"], "is_whiff", None, f_swing["is_whiff"].mean(), W_SWINGS / 2, "x")
        out[f"prev_season_hardhit_rate_{fam}"] = _rate_block(
            f_bip, ["batter"], "is_hard", None, f_bip["is_hard"].mean(), W_BIP / 2, "x")
        e = f_bip.dropna(subset=["launch_speed"]).groupby("batter")["launch_speed"].agg(["sum", "count"])
        out[f"prev_season_ev_{fam}"] = shrink(e["sum"], e["count"], f_bip["launch_speed"].mean(), W_BIP / 2)

        # -- 各球種族 x 區域：揮棒率、揮空率 --
        for reg in REGIONS:
            cell = f_all[f_all["region"] == reg]
            cell_sw = cell[cell["is_swing"]]
            out[f"prev_season_swing_rate_{fam}_{reg}"] = _rate_block(
                cell, ["batter"], "is_swing", None, cell["is_swing"].mean(), W_PITCHES / 4, "x")
            out[f"prev_season_whiff_rate_{fam}_{reg}"] = _rate_block(
                cell_sw, ["batter"], "is_whiff", None, cell_sw["is_whiff"].mean(), W_SWINGS / 4, "x")

    # 某打者完全沒遇過某一格時，收縮公式的分子分母都不存在 -> 用該格聯盟值
    # (沒有 bat tracking 的年份，聯盟值本身就是 NaN，填補後仍為 NaN)
    league_row = _league_row(df, out.columns, swings, bip, chase, spray, has_bat_tracking)
    out = out.fillna({c: league_row[c] for c in out.columns if pd.notna(league_row[c])})
    out["prev_season_n_swings"] = out["prev_season_n_swings"].fillna(0)

    league_df = pd.DataFrame([league_row], index=pd.Index([LEAGUE_ROW_ID], name="batter")).infer_objects()
    out = pd.concat([out, league_df.astype(out.dtypes.to_dict())])
    out["profile_season"] = season
    out["source_season"] = src_year
    return out.reset_index()


def _league_row(df, columns, swings, bip, chase, spray, has_bat_tracking) -> pd.Series:
    """聯盟平均列：冷啟動打者與空格的填補值。"""
    row = pd.Series(index=columns, dtype="object")
    row["prev_season_n_pitches"] = 0
    row["prev_season_n_swings"] = 0
    row["prev_season_swing_rate"] = df["is_swing"].mean()
    row["prev_season_whiff_rate"] = swings["is_whiff"].mean()
    row["prev_season_chase_rate"] = chase["is_swing"].mean()
    row["prev_season_bip_rate"] = df["is_bip"].mean()
    row["prev_season_hardhit_rate"] = bip["is_hard"].mean()
    row["prev_season_pull_rate"] = spray["is_pulled"].mean()
    row["prev_season_ev"] = bip["launch_speed"].mean()
    row["bat_tracking_available"] = has_bat_tracking
    for reg in ("high", "mid", "low"):
        row[f"prev_season_hardhit_rate_{reg}"] = bip.loc[bip["region"] == reg, "is_hard"].mean()
    if has_bat_tracking:
        tracked = swings.dropna(subset=["bat_speed"])
        row["prev_season_mean_bat_speed"] = tracked["bat_speed"].mean()
        row["prev_season_mean_swing_length"] = tracked["swing_length"].mean()
        row["prev_season_mean_attack_angle"] = tracked["attack_angle"].mean()
        row["prev_season_fast_swing_rate"] = (tracked["bat_speed"] >= FAST_SWING_MPH).mean()
    for fam in FAMILIES:
        f_all = df[df["pitch_family"] == fam]
        f_sw, f_bip, f_ch = f_all[f_all["is_swing"]], f_all[f_all["is_bip"]], f_all[f_all["region"] == "chase"]
        row[f"prev_season_chase_rate_{fam}"] = f_ch["is_swing"].mean()
        row[f"prev_season_whiff_rate_{fam}"] = f_sw["is_whiff"].mean()
        row[f"prev_season_hardhit_rate_{fam}"] = f_bip["is_hard"].mean()
        row[f"prev_season_ev_{fam}"] = f_bip["launch_speed"].mean()
        for reg in REGIONS:
            cell = f_all[f_all["region"] == reg]
            row[f"prev_season_swing_rate_{fam}_{reg}"] = cell["is_swing"].mean()
            row[f"prev_season_whiff_rate_{fam}_{reg}"] = cell.loc[cell["is_swing"], "is_whiff"].mean()
    return row


# ---------------------------------------------------------------------------
# 2. 近期輪廓 (滾動 30 天拉打率)
# ---------------------------------------------------------------------------

def build_recent(season: int, prev: pd.DataFrame | None) -> pd.DataFrame | None:
    df = load_interim(season)
    if df is None:
        return None

    df["date"] = pd.to_datetime(df["game_date"])
    agg = {"swings": ("is_swing", "sum"), "spray_n": ("has_spray", "sum"), "pulled": ("is_pulled", "sum")}
    daily = df.groupby(["batter", "date"]).agg(**agg)
    for fam in FAMILIES:
        f = df[df["pitch_family"] == fam].groupby(["batter", "date"]).agg(
            **{f"spray_n_{fam}": ("has_spray", "sum"), f"pulled_{fam}": ("is_pulled", "sum")})
        daily = daily.join(f, how="left")
    daily = daily.fillna(0.0).reset_index().sort_values(["batter", "date"])

    # 窗口 [date - 30 天, date)：closed="left" 排除比賽當天
    sum_cols = [c for c in daily.columns if c not in ("batter", "date")]
    rolled = (daily.set_index("date").groupby("batter")[sum_cols]
                   .rolling(RECENT_WINDOW, closed="left").sum()
                   .reset_index())
    rolled[sum_cols] = rolled[sum_cols].fillna(0.0)

    # 退路：上一季拉打率 (新人或沒有上一季資料時用聯盟值)
    if prev is not None:
        prev_pull = prev.set_index("batter")["prev_season_pull_rate"]
        league_pull = float(prev_pull.loc[LEAGUE_ROW_ID])
        fallback = rolled["batter"].map(prev_pull.drop(LEAGUE_ROW_ID)).fillna(league_pull)
        fallback_source = np.where(rolled["batter"].isin(prev_pull.index.drop(LEAGUE_ROW_ID)),
                                   "prev_season", "league_prev_season")
    else:
        # 沒有上一季檔案時，只能用本季聯盟值當常數先驗。這是群體層級的常數，不含個別打者資訊。
        league_pull = float(df.loc[df["has_spray"], "is_pulled"].mean())
        fallback = pd.Series(league_pull, index=rolled.index)
        fallback_source = np.full(len(rolled), "league_same_season")

    enough = rolled["swings"] >= RECENT_MIN_SWINGS
    blended = shrink(rolled["pulled"], rolled["spray_n"], fallback, W_RECENT)
    out = rolled[["batter", "date", "swings", "spray_n"]].rename(
        columns={"swings": "recent_n_swings", "spray_n": "recent_n_batted"})
    out["recent_pull_rate"] = np.where(enough, blended, fallback)
    out["recent_window_sufficient"] = enough
    out["recent_fallback_source"] = fallback_source
    for fam in FAMILIES:
        fam_rate = shrink(rolled[f"pulled_{fam}"], rolled[f"spray_n_{fam}"], out["recent_pull_rate"], W_RECENT)
        out[f"recent_pull_rate_{fam}"] = np.where(enough, fam_rate, out["recent_pull_rate"])

    out["game_date"] = out.pop("date").dt.strftime("%Y-%m-%d")
    return out


# ---------------------------------------------------------------------------
# 3. 合併 (供建模階段使用)
# ---------------------------------------------------------------------------

def attach_batter_profile(pitches: pd.DataFrame, season: int) -> pd.DataFrame:
    """把兩份輪廓合併到逐球資料上。冷啟動打者填聯盟平均並標記 is_rookie。"""
    prev_path = spec.INTERIM_DIR / f"batter_prev_season_{season}.parquet"
    recent_path = spec.INTERIM_DIR / f"batter_recent_{season}.parquet"
    out = pitches

    if prev_path.exists():
        prev = pd.read_parquet(prev_path)
        league = prev[prev["batter"] == LEAGUE_ROW_ID].iloc[0]
        cols = [c for c in prev.columns if c.startswith("prev_season_")]
        out = out.merge(prev[prev["batter"] != LEAGUE_ROW_ID][["batter"] + cols], on="batter", how="left")
        out["is_rookie"] = out["prev_season_n_pitches"].isna()
        out[cols] = out[cols].fillna(league[cols])
    else:
        out = out.assign(is_rookie=True)

    if recent_path.exists():
        recent = pd.read_parquet(recent_path)
        out = out.merge(recent, on=["batter", "game_date"], how="left")
        # 當天所有的球都不是建模目標 (例如整天只有觸擊打席) 時，近期表沒有這一天 -> 退回上一季值
        missing = out["recent_pull_rate"].isna()
        if missing.any():
            base = out["prev_season_pull_rate"] if "prev_season_pull_rate" in out else np.nan
            rate_cols = ["recent_pull_rate"] + [f"recent_pull_rate_{f}" for f in FAMILIES]
            for c in rate_cols:
                out.loc[missing, c] = base[missing] if isinstance(base, pd.Series) else base
            out.loc[missing, ["recent_n_swings", "recent_n_batted"]] = 0.0
            out.loc[missing, "recent_window_sufficient"] = False
            out.loc[missing, "recent_fallback_source"] = "no_recent_row"
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_season(season: int) -> bool:
    spec.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    done = False

    prev = build_prev_season(season)
    if prev is None:
        log(f"{season}: 沒有 pitches_{season - 1}.parquet，無法建立上一季輪廓 (該季打者全部視為冷啟動)")
    else:
        path = spec.INTERIM_DIR / f"batter_prev_season_{season}.parquet"
        prev.to_parquet(path, index=False)
        n = (prev["batter"] != LEAGUE_ROW_ID).sum()
        log(f"{season}: 上一季輪廓 (來源 {season - 1}) {n:,} 位打者, {prev.shape[1]} 欄 -> {path.name} "
            f"| bat tracking: {'有' if prev['bat_tracking_available'].iloc[0] else '無'}")
        done = True

    recent = build_recent(season, prev)
    if recent is None:
        log(f"{season}: 沒有 pitches_{season}.parquet，跳過近期輪廓")
    else:
        path = spec.INTERIM_DIR / f"batter_recent_{season}.parquet"
        recent.to_parquet(path, index=False)
        log(f"{season}: 近期輪廓 {len(recent):,} 列 (打者 x 比賽日) -> {path.name} "
            f"| 窗口樣本足夠 {recent['recent_window_sufficient'].mean():.1%} "
            f"| 退路來源 {recent['recent_fallback_source'].value_counts(normalize=True).round(3).to_dict()}")
        done = True
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="建立打者輪廓 (上一季 / 近 30 天)")
    ap.add_argument("--season", default="all", help="要使用輪廓的賽季：年份 / 區間 / 逗號列舉 / all")
    args = ap.parse_args()
    results = [run_season(int(y)) for y in resolve_seasons(args.season)]
    return 0 if any(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
