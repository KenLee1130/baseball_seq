"""
splits.py
=========
依「球季」切分 train / val / test，並定義哪些球可以當預測目標。

為什麼用球季切，不用隨機切：
  - 打者輪廓用「上一季」，隨機切會讓同一季的球同時出現在 train 與 test，
    模型可以從 train 裡學到該打者當季的狀態，再拿去預測 test，結果虛高。
  - 配球策略會隨年代演變 (sweeper 普及、投球計時器)，時間切分才能回答
    「用過去訓練的模型，能不能預測未來」。

切分設定 (SPLITS)：
  train    2016-2023   2015 沒有上一季 (2014 未下載)，打者輪廓全是冷啟動，故不納入
  val      2024        選超參數、early stopping
  test     2025-2026   最終評估，只在定案後看一次

本檔案不複製資料：splits 只是「哪一季屬於哪一份 + 哪些球可當目標」的規則。
實際讀取用 load_season() / iter_split()，每次從 interim 讀並合併打者輪廓。

用法:
    python data_preparation/splits.py            # 檢查所有球季、寫出 manifest 與分布比較
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_spec as spec
from batter_profile import attach_batter_profile

SPLITS: dict[str, list[int]] = {
    "train": list(range(2016, 2024)),
    "val": [2024],
    "test": [2025, 2026],
}

# 2022 以前國聯投手要上場打擊。投手打擊的反應和一般打者完全不同，不當預測目標；
# 但他們面對的球仍保留在序列中 (那是投手配球脈絡的一部分)。
EXCLUDE_PITCHER_BATTING = True

MANIFEST_PATH = spec.SPLIT_DIR / "manifest.json"

# load_season 自己需要的欄位 (排序、合併輪廓、判斷目標)
REQUIRED_COLUMNS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "is_model_target"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def split_of(season: int) -> str | None:
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    return None


def _pitcher_ids() -> set[int]:
    bio = pd.read_parquet(spec.RAW_DIR / "player_bio.parquet", columns=["player_id", "primary_position"])
    # 二刀流 (TWP，例如大谷翔平) 不算投手打擊
    return set(bio.loc[bio["primary_position"] == "P", "player_id"].astype(int))


def load_season(season: int, columns: list[str] | None = None) -> pd.DataFrame:
    """讀一季逐球資料，合併打者輪廓，並標記可當預測目標的球。

    保留所有列 (含不可當目標的球)，因為序列需要完整的前後文。
    """
    path = spec.INTERIM_DIR / f"pitches_{season}.parquet"
    if columns is not None:
        # 呼叫端可以一併列出打者輪廓欄位；那些欄位不在逐球檔裡，合併時才會出現
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [c for c in dict.fromkeys(REQUIRED_COLUMNS + list(columns)) if c in available]
    df = pd.read_parquet(path, columns=columns)
    df = attach_batter_profile(df, season)
    df["season"] = season
    df["split"] = split_of(season)

    has_prev_profile = (spec.INTERIM_DIR / f"batter_prev_season_{season}.parquet").exists()
    target = df["is_model_target"] & has_prev_profile
    if EXCLUDE_PITCHER_BATTING:
        df["batter_is_pitcher"] = df["batter"].astype(int).isin(_pitcher_ids())
        target &= ~df["batter_is_pitcher"]
    df["use_as_target"] = target
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


def iter_split(name: str, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    """逐季讀取某一份切分。不一次載入全部，避免 8 季 x 70 萬球同時佔記憶體。"""
    for season in SPLITS[name]:
        yield load_season(season, columns)


# ---------------------------------------------------------------------------
# 檢查與 manifest
# ---------------------------------------------------------------------------

def check_splits() -> None:
    """切分規則本身的一致性：季別不重疊、每季的打者輪廓只來自前一季。"""
    seen: dict[int, str] = {}
    for name, seasons in SPLITS.items():
        for s in seasons:
            if s in seen:
                raise SystemExit(f"球季 {s} 同時屬於 {seen[s]} 與 {name}")
            seen[s] = name
    names = list(SPLITS)
    for earlier, later in zip(names, names[1:]):
        if max(SPLITS[earlier]) >= min(SPLITS[later]):
            raise SystemExit(f"切分必須依時間先後：{earlier} 的球季必須全部早於 {later}")

    for s in seen:
        path = spec.INTERIM_DIR / f"batter_prev_season_{s}.parquet"
        if path.exists():
            src = pd.read_parquet(path, columns=["source_season"])["source_season"].unique()
            if list(src) != [s - 1]:
                raise SystemExit(f"{path.name} 的來源球季是 {src}，應該只有 {s - 1}")


SUMMARY_COLUMNS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "is_model_target",
                   "is_swing", "is_contact", "ev_measured", "contact3", "pitch_family", "count_state"]


def season_summary(season: int) -> dict:
    df = load_season(season, SUMMARY_COLUMNS)
    t = df[df["use_as_target"]]
    swings = t[t["is_swing"]]
    c3 = t.loc[t["contact3"] >= 0, "contact3"].value_counts(normalize=True)
    return {
        "season": season,
        "split": split_of(season),
        "n_pitches": int(len(df)),
        "n_targets": int(len(t)),
        "target_share": round(len(t) / len(df), 4),
        "pitcher_batting_share": round(float(df.get("batter_is_pitcher", pd.Series(False)).mean()), 4),
        "rookie_share": round(float(t["is_rookie"].mean()), 4),
        "swing_rate": round(float(t["is_swing"].mean()), 4),
        "contact_rate_given_swing": round(float(swings["is_contact"].mean()), 4),
        "ev_n": int(t["ev_measured"].notna().sum()),
        "ev_mean": round(float(t["ev_measured"].mean()), 2),
        "contact3_none": round(float(c3.get(0, 0)), 4),
        "contact3_soft": round(float(c3.get(1, 0)), 4),
        "contact3_hard": round(float(c3.get(2, 0)), 4),
        "family_share": {k: round(float(v), 4) for k, v in t["pitch_family"].value_counts(normalize=True).items()},
    }


def main() -> int:
    argparse.ArgumentParser(description="檢查切分並輸出 manifest").parse_args()
    check_splits()
    log("切分規則檢查通過：季別不重疊、依時間先後、打者輪廓只來自前一季")

    rows = []
    for name, seasons in SPLITS.items():
        for s in seasons:
            if not (spec.INTERIM_DIR / f"pitches_{s}.parquet").exists():
                log(f"! 缺 pitches_{s}.parquet，先執行 preprocess.py")
                continue
            rows.append(season_summary(s))
            r = rows[-1]
            log(f"  {s} [{name:<7}] 目標 {r['n_targets']:>8,} ({r['target_share']:.1%}) | 新人 {r['rookie_share']:.1%} "
                f"| 出棒 {r['swing_rate']:.1%} | 觸球|揮棒 {r['contact_rate_given_swing']:.1%} "
                f"| 強擊 {r['contact3_hard']:.1%} | 投手打擊 {r['pitcher_batting_share']:.1%}")

    table = pd.DataFrame(rows)
    num = ["n_targets", "rookie_share", "swing_rate", "contact_rate_given_swing", "ev_mean",
           "contact3_none", "contact3_soft", "contact3_hard"]
    by_split = table.groupby("split", sort=False).apply(
        lambda g: pd.Series({c: (g[c].sum() if c == "n_targets" else np.average(g[c], weights=g["n_targets"]))
                             for c in num}), include_groups=False)
    log("各切分的分布比較 (以目標球數加權)：\n" + by_split.round(4).to_string())

    spec.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": SPLITS, "exclude_pitcher_batting": EXCLUDE_PITCHER_BATTING,
                "generated_at": datetime.now().isoformat(timespec="seconds"), "seasons": rows}
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"-> {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
