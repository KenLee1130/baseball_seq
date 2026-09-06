"""
fetch_data.py
=============
照 data_spec.py 的規格抓原始資料，存進 dataset/raw/。

本檔案只做「抓」與「最低限度的過濾」，不做任何特徵工程、不做正規化、
不切 train/test。那些屬於 preprocess 階段。這樣重跑時不必重抓網路資料。

抓兩種東西：
  1. Statcast 逐球資料 (pybaseball -> Baseball Savant)
  2. 球員身高體重 (MLB Stats API，因為 Statcast 沒有這些欄位)

設計重點：
  - 逐月分塊下載，每塊各存一份 parquet。中斷後重跑會跳過已完成的塊。
  - 只做兩件過濾：非例行賽、非競技投球 (牽制、故意四壞)。
    其餘過濾 (觸擊、揮棒母體) 留到 preprocess，因為那牽涉建模決策。

用法:
    python fetch_data.py --smoke              # 3 天資料，驗證管線
    python fetch_data.py --season model       # 只抓 2025
    python fetch_data.py --season all         # 抓 2024 + 2025 (約 150 萬球)
    python fetch_data.py --season all --force # 忽略既有檔案重抓
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_spec as spec

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ensure_dirs() -> None:
    for d in (spec.RAW_DIR, spec.INTERIM_DIR, spec.SPLIT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def month_chunks(start: str, end: str) -> list[tuple[str, str, str]]:
    """把日期區間切成逐月的塊，回傳 (label, start, end)。

    逐月而非整季一次抓的理由：Savant 對大區間查詢容易逾時，
    且分塊後中斷可續傳。
    """
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    chunks: list[tuple[str, str, str]] = []
    cur = s
    while cur <= e:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), e)
        chunks.append((f"{cur:%Y-%m}", cur.isoformat(), chunk_end.isoformat()))
        cur = nxt
    return chunks


# ---------------------------------------------------------------------------
# 1. Statcast 逐球資料
# ---------------------------------------------------------------------------

def _select_spec_columns(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 data_spec 列出的欄位，並回報缺漏。

    Savant 偶爾改動欄位。缺漏要明確報出來，不能默默吞掉。
    """
    wanted = spec.ALL_STATCAST_COLUMNS
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        log(f"  ! Savant 未回傳這些規格欄位: {missing}")
    return df[present].copy()


def _basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """最低限度過濾。建模相關的過濾留給 preprocess。"""
    n0 = len(df)

    if "game_type" in df.columns:
        df = df[df["game_type"].isin(spec.KEEP_GAME_TYPES)]
    n1 = len(df)

    # 牽制、故意四壞等非競技投球
    df = df[~df["pitch_type"].isin(spec.EXCLUDE_PITCH_TYPES)]
    df = df[df["pitch_type"].notna()]
    n2 = len(df)

    # pitchout / 計時違規的自動好壞球：不是打者對球的判斷
    df = df[~df["description"].isin(spec.NON_COMPETITIVE_DESCRIPTIONS)]
    n3 = len(df)

    log(f"  過濾: {n0} -> 例行賽 {n1} -> 競技球種 {n2} -> 競技結果 {n3}")
    return df


def fetch_statcast_chunk(label: str, start: str, end: str, force: bool) -> Path | None:
    """抓一個月的逐球資料，存成 parquet。已存在則跳過。"""
    out = spec.RAW_DIR / f"statcast_{label}.parquet"
    if out.exists() and not force:
        log(f"  跳過 {label} (已存在, {out.stat().st_size / 1e6:.1f} MB)")
        return out

    from pybaseball import statcast

    log(f"  下載 {label} ({start} ~ {end}) ...")
    t0 = time.time()
    try:
        df = statcast(start_dt=start, end_dt=end, verbose=False)
    except Exception as exc:  # noqa: BLE001
        log(f"  ! {label} 下載失敗: {exc}")
        return None

    if df is None or df.empty:
        log(f"  {label} 無資料 (可能是休賽期)，跳過")
        return None

    df = _basic_filter(df)
    df = _select_spec_columns(df)

    # 序列骨架的排序：同一打席內必須依 pitch_number 遞增
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    df.to_parquet(out, index=False)
    log(f"  {label}: {len(df):,} 球, {time.time() - t0:.0f}s, -> {out.name}")
    return out


def fetch_statcast_season(season_key: str, force: bool) -> pd.DataFrame:
    """抓整季，回傳合併後的 DataFrame。"""
    cfg = spec.SEASONS[season_key]
    log(f"=== Statcast {season_key} 賽季 {cfg['year']} ({cfg['start']} ~ {cfg['end']}) ===")

    paths: list[Path] = []
    for label, s, e in month_chunks(cfg["start"], cfg["end"]):
        p = fetch_statcast_chunk(label, s, e, force)
        if p is not None:
            paths.append(p)

    if not paths:
        raise RuntimeError(f"{season_key} 賽季沒有抓到任何資料")

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    out = spec.RAW_DIR / f"statcast_{cfg['year']}.parquet"
    df.to_parquet(out, index=False)
    log(f"=== {cfg['year']} 合併完成: {len(df):,} 球, {df.game_pk.nunique():,} 場 -> {out.name} ===")
    return df


# ---------------------------------------------------------------------------
# 2. 球員身高體重 (MLB Stats API)
# ---------------------------------------------------------------------------

_HEIGHT_RE_NOTE = "MLB API 的 height 是 \"5' 11\\\"\" 這種字串，要自己轉成吋"


def _parse_height_to_inches(h: str | None) -> float | None:
    """把 "5' 11\"" 轉成 71.0 吋。"""
    if not h or not isinstance(h, str):
        return None
    try:
        parts = h.replace('"', "").split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return float(feet * 12 + inches)
    except (ValueError, IndexError):
        return None


def fetch_player_bio(player_ids: list[int], force: bool = False) -> pd.DataFrame:
    """批次抓球員身高、體重、慣用邊。

    Statcast 沒有身高體重，必須另外從 MLB Stats API 取得。
    該端點支援一次查詢多個 personIds，所以不需要一人一個請求。
    """
    out = spec.RAW_DIR / "player_bio.parquet"
    if out.exists() and not force:
        existing = pd.read_parquet(out)
        missing = sorted(set(player_ids) - set(existing["player_id"]))
        if not missing:
            log(f"球員資料已完整 ({len(existing)} 人)，跳過")
            return existing
        log(f"球員資料缺 {len(missing)} 人，補抓")
        player_ids = missing
    else:
        existing = pd.DataFrame()

    url = spec.EXTERNAL_SOURCES["mlb_stats_api_people"]["url"]
    rows: list[dict] = []
    batch_size = 100

    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i : i + batch_size]
        try:
            resp = requests.get(
                url, params={"personIds": ",".join(map(str, batch))}, timeout=30
            )
            resp.raise_for_status()
            people = resp.json().get("people", [])
        except Exception as exc:  # noqa: BLE001
            log(f"  ! 球員批次 {i // batch_size} 失敗: {exc}")
            continue

        for p in people:
            rows.append(
                {
                    "player_id": p.get("id"),
                    "full_name": p.get("fullName"),
                    "height_raw": p.get("height"),
                    "height_in": _parse_height_to_inches(p.get("height")),
                    "weight_lb": p.get("weight"),
                    "bats": (p.get("batSide") or {}).get("code"),
                    "throws": (p.get("pitchHand") or {}).get("code"),
                    "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
                    "birth_date": p.get("birthDate"),
                }
            )
        log(f"  球員 {i + len(batch)}/{len(player_ids)}")
        time.sleep(0.3)  # 對公開 API 客氣一點

    df = pd.DataFrame(rows)
    if not existing.empty:
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("player_id")

    df.to_parquet(out, index=False)
    log(f"球員資料: {len(df)} 人 -> {out.name}")
    return df


# ---------------------------------------------------------------------------
# 3. 抓完之後的完整性檢查
# ---------------------------------------------------------------------------

def sanity_report(df: pd.DataFrame) -> None:
    """印出關鍵欄位的可用率。抓完一定要看，因為缺值率決定建模母體大小。"""
    log("--- 完整性檢查 ---")
    log(f"  總球數 {len(df):,} / 場次 {df.game_pk.nunique():,} / "
        f"打者 {df.batter.nunique():,} / 投手 {df.pitcher.nunique():,}")

    desc = df["description"]
    n_swing = desc.isin(spec.SWING_DESCRIPTIONS).sum()
    n_contact = desc.isin(spec.CONTACT_DESCRIPTIONS).sum()
    n_ev = df.loc[desc.isin(spec.EV_MEASURED_DESCRIPTIONS), "launch_speed"].notna().sum()
    log(f"  Stage1 母體 {len(df):,} 球，出棒 {n_swing:,} ({n_swing / len(df):.1%})")
    log(f"  Stage2 母體 {n_swing:,} 揮棒，觸球 {n_contact:,} ({n_contact / max(n_swing,1):.1%})")
    log(f"  Stage3 母體 {n_ev:,} 球有擊球初速 ({n_ev / max(n_contact,1):.1%} of 觸球)")

    key_cols = [
        "plate_x", "plate_z", "release_speed", "release_spin_rate",
        "pfx_x", "release_pos_x", "release_extension", "arm_angle",
        "bat_speed", "swing_length", "attack_angle",
        "launch_speed", "hc_x", "delta_run_exp",
    ]
    log("  關鍵欄位可用率:")
    for c in key_cols:
        if c in df.columns:
            log(f"    {c:<22} {df[c].notna().mean():>6.1%}")
        else:
            log(f"    {c:<22} 欄位不存在")

    # 序列骨架必須唯一，否則 shift 出來的「前一球」會錯
    dup = df.duplicated(subset=["game_pk", "at_bat_number", "pitch_number"]).sum()
    log(f"  序列主鍵重複數: {dup} (必須為 0)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="抓 Statcast 逐球資料與球員身高體重")
    ap.add_argument(
        "--season", default="model", choices=["profile", "model", "all"],
        help="profile=2024 分群用, model=2025 建模用, all=兩季都抓",
    )
    ap.add_argument("--force", action="store_true", help="忽略既有檔案，重新下載")
    ap.add_argument("--smoke", action="store_true", help="只抓 3 天資料，驗證管線")
    ap.add_argument("--skip-bio", action="store_true", help="不抓球員身高體重")
    args = ap.parse_args()

    ensure_dirs()

    try:
        from pybaseball import cache
        cache.enable()
        log("pybaseball 快取已啟用")
    except Exception:  # noqa: BLE001
        log("pybaseball 快取無法啟用，繼續")

    if args.smoke:
        log("=== SMOKE TEST: 2025-06-01 ~ 2025-06-03 ===")
        p = fetch_statcast_chunk("smoke", "2025-06-01", "2025-06-03", force=True)
        if p is None:
            return 1
        df = pd.read_parquet(p)
        sanity_report(df)
        if not args.skip_bio:
            ids = sorted(set(df["batter"].dropna().astype(int)))[:100]
            fetch_player_bio(ids, force=args.force)
        log("SMOKE TEST 完成")
        return 0

    keys = ["profile", "model"] if args.season == "all" else [args.season]
    frames: list[pd.DataFrame] = []
    for k in keys:
        frames.append(fetch_statcast_season(k, args.force))

    combined = pd.concat(frames, ignore_index=True)
    sanity_report(combined)

    if not args.skip_bio:
        ids = sorted(
            set(combined["batter"].dropna().astype(int))
            | set(combined["pitcher"].dropna().astype(int))
        )
        log(f"需要 {len(ids)} 位球員的身高體重")
        fetch_player_bio(ids, force=args.force)

    log("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
