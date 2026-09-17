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

不重複下載的三層機制 (抓 2015-2026 共 12 季時，這點決定能不能重跑)：
  1. 已定版的月檔直接跳過。「定版」= 該月已結束，且檔案寫於該月結束之後。
     只檢查檔案存在是不夠的：賽季進行中抓下來的當月檔案只有半個月的球，
     之後必須重抓，否則那個月永遠殘缺。
  2. 確認無比賽的月份寫下 .empty 標記，之後不再向 Savant 查詢。
     休賽期的月份在 12 季裡有數十個，每次都問一遍很浪費。
  3. 年檔只在該季的月檔真的有變動時才重新合併。

用法:
    python fetch_data.py --smoke                # 3 天資料，驗證管線
    python fetch_data.py --season all           # 2015-2026 全部 (預設)
    python fetch_data.py --season 2019          # 單季
    python fetch_data.py --season 2015-2020     # 區間
    python fetch_data.py --season 2024,2025     # 列舉
    python fetch_data.py --season all --report  # 每季印完整性檢查
    python fetch_data.py --season 2020 --force  # 忽略既有檔案重抓該季
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


def _statcast_range(start: str, end: str, retries: int = 2) -> pd.DataFrame | None:
    """抓一個日期區間，失敗時退化成逐日抓。

    為什麼需要退化：實測 2015-04 整月查詢會拋
    "Error tokenizing data. C error: Expected 1 fields in line 12, saw 2"，
    但那個月裡每一天單獨抓都成功。表示問題出在 Savant 對大區間的回應
    偶發截斷，不是資料本身缺失。整季直接放棄太可惜，逐日重試即可救回。

    逐日模式慢得多 (一個月 30 次請求)，所以只在整區間失敗後才啟用。
    """
    from pybaseball import statcast

    for attempt in range(retries):
        try:
            return statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception as exc:  # noqa: BLE001
            log(f"    區間查詢失敗 (第 {attempt + 1}/{retries} 次): {str(exc)[:80]}")
            time.sleep(2 * (attempt + 1))

    log("    改為逐日抓取")
    cur = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    while cur <= stop:
        day = cur.isoformat()
        try:
            d = statcast(start_dt=day, end_dt=day, verbose=False)
            if d is not None and not d.empty:
                frames.append(d)
        except Exception:  # noqa: BLE001
            failed.append(day)
        cur += timedelta(days=1)

    if failed:
        # 明確報出來。靜靜吞掉缺天會讓序列特徵在該日斷裂而不自知。
        log(f"    ! 逐日模式仍有 {len(failed)} 天失敗: {failed[:5]}{' ...' if len(failed) > 5 else ''}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _chunk_is_fresh(out: Path, chunk_end: str, today: date) -> bool:
    """判斷既有的月檔是否已經是「定版」，可以安全跳過。

    只看檔案存在是不夠的：若某個月在賽季進行中就抓下來，
    當時只有到抓取當天為止的比賽，之後那個月又打了半個月的球。
    直接跳過會讓那個月永遠殘缺。

    規則：該月已經結束 (chunk_end < today)，且檔案是在該月結束之後才寫入的，
    才算定版。否則視為過期，重抓。
    """
    if not out.exists():
        return False
    end = date.fromisoformat(chunk_end)
    if end >= today:
        return False  # 這個月還沒過完，資料必然不完整
    written = date.fromtimestamp(out.stat().st_mtime)
    return written > end


def fetch_statcast_chunk(
    label: str, start: str, end: str, force: bool, today: date | None = None
) -> Path | None:
    """抓一個月的逐球資料，存成 parquet。

    跳過條件有兩種，都是為了「重複的不要下載」：
      1. 已有定版的月檔 (見 _chunk_is_fresh)
      2. 已有 .empty 標記，代表該月確認無比賽 (休賽期)，不必再問 Savant 一次
    """
    today = today or date.today()
    out = spec.RAW_DIR / f"statcast_{label}.parquet"
    empty_marker = spec.RAW_DIR / f"statcast_{label}.empty"

    if not force:
        if _chunk_is_fresh(out, end, today):
            log(f"  跳過 {label} (已定版, {out.stat().st_size / 1e6:.1f} MB)")
            return out
        if empty_marker.exists():
            log(f"  跳過 {label} (已知無比賽)")
            return None
        if out.exists():
            log(f"  重抓 {label} (既有檔案可能殘缺: 抓取時該月尚未結束)")

    log(f"  下載 {label} ({start} ~ {end}) ...")
    t0 = time.time()
    df = _statcast_range(start, end)

    if df is None or df.empty:
        # 只有當這個月已經完全過去，「沒有比賽」才是個永久事實，才值得標記。
        if date.fromisoformat(end) < today:
            empty_marker.write_text(f"no games {start}..{end}\n")
            log(f"  {label} 無資料 (休賽期)，寫下標記，之後不再查詢")
        else:
            log(f"  {label} 無資料 (尚未開打)，不標記")
        return None

    df = _basic_filter(df)
    df = _select_spec_columns(df)

    # 序列骨架的排序：同一打席內必須依 pitch_number 遞增
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    df.to_parquet(out, index=False)
    empty_marker.unlink(missing_ok=True)  # 之前若誤標為空，現在有資料了
    log(f"  {label}: {len(df):,} 球, {time.time() - t0:.0f}s, -> {out.name}")
    return out


def fetch_statcast_season(season_key: str, force: bool) -> Path | None:
    """抓整季，合併成年檔，回傳年檔路徑。

    刻意回傳路徑而非 DataFrame：抓 2015-2026 共 12 季時，
    把每一季都留在記憶體裡會用掉數 GB。呼叫端需要時再自己讀。
    """
    cfg = spec.SEASONS[season_key]
    year = cfg["year"]
    log(f"=== Statcast {year} ({cfg['start']} ~ {cfg['end']}) ===")

    today = date.today()
    out = spec.RAW_DIR / f"statcast_{year}.parquet"

    paths: list[Path] = []
    rebuilt = False
    for label, s, e in month_chunks(cfg["start"], cfg["end"]):
        before = out.exists() and (spec.RAW_DIR / f"statcast_{label}.parquet").exists()
        mtime_before = (
            (spec.RAW_DIR / f"statcast_{label}.parquet").stat().st_mtime if before else None
        )
        pth = fetch_statcast_chunk(label, s, e, force, today)
        if pth is not None:
            paths.append(pth)
            if mtime_before is None or pth.stat().st_mtime != mtime_before:
                rebuilt = True

    if not paths:
        log(f"  {year} 沒有任何月份有資料，跳過")
        return None

    # 年檔只在月檔真的有變動時才重建。否則 12 季每次都重寫近 100 MB 很浪費。
    if out.exists() and not rebuilt and not force:
        log(f"=== {year} 年檔已是最新，跳過合併 ({out.stat().st_size / 1e6:.0f} MB) ===")
        return out

    df = pd.concat([pd.read_parquet(pth) for pth in paths], ignore_index=True)
    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)
    df.to_parquet(out, index=False)
    log(f"=== {year} 合併完成: {len(df):,} 球, {df.game_pk.nunique():,} 場 -> {out.name} ===")
    return out


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

def resolve_seasons(arg: str) -> list[str]:
    """把 --season 的值解析成年份清單。

    支援四種寫法，因為抓 12 季時常常只想補其中幾季：
        all          全部 (2015-2026)
        2019         單季
        2019-2022    連續區間 (含頭含尾)
        2019,2023    逗號列舉
        profile/model 語意別名 (見 data_spec.SEASON_ALIASES)
    """
    arg = arg.strip()
    if arg == "all":
        return sorted(spec.SEASONS)
    if arg in spec.SEASON_ALIASES:
        return [spec.SEASON_ALIASES[arg]]

    out: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if part in spec.SEASON_ALIASES:
            out.append(spec.SEASON_ALIASES[part])
        elif "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(str(y) for y in range(int(lo), int(hi) + 1))
        else:
            out.append(part)

    unknown = [y for y in out if y not in spec.SEASONS]
    if unknown:
        raise SystemExit(
            f"未知的賽季 {unknown}；可用範圍 "
            f"{spec.FIRST_STATCAST_YEAR}-{spec.LAST_SEASON_YEAR}"
        )
    return sorted(dict.fromkeys(out))


def collect_player_ids(paths: list[Path]) -> list[int]:
    """從年檔收集所有出現過的球員 ID。

    逐檔只讀 batter / pitcher 兩欄再取聯集，不把整份資料讀進記憶體。
    12 季全讀是 7 GB 起跳，只讀兩欄是幾十 MB。
    """
    ids: set[int] = set()
    for pth in paths:
        df = pd.read_parquet(pth, columns=["batter", "pitcher"])
        ids |= set(df["batter"].dropna().astype(int))
        ids |= set(df["pitcher"].dropna().astype(int))
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="抓 Statcast 逐球資料與球員身高體重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  --season all          全部 2015-2026\n"
            "  --season 2019         只抓 2019\n"
            "  --season 2015-2020    抓 2015 到 2020\n"
            "  --season 2024,2025    只抓這兩季\n"
            "已下載且已定版的月份會自動跳過，重跑安全。"
        ),
    )
    ap.add_argument(
        "--season", default="all",
        help="年份 / 區間 / 逗號列舉 / all / profile / model (預設 all)",
    )
    ap.add_argument("--force", action="store_true", help="忽略既有檔案，重新下載")
    ap.add_argument("--smoke", action="store_true", help="只抓 3 天資料，驗證管線")
    ap.add_argument("--skip-bio", action="store_true", help="不抓球員身高體重")
    ap.add_argument(
        "--report", action="store_true",
        help="每季抓完後印出完整性檢查 (需把整季讀進記憶體，約 600 MB/季)",
    )
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
        pth = fetch_statcast_chunk("smoke", "2025-06-01", "2025-06-03", force=True)
        if pth is None:
            return 1
        df = pd.read_parquet(pth)
        sanity_report(df)
        if not args.skip_bio:
            ids = sorted(set(df["batter"].dropna().astype(int)))[:100]
            fetch_player_bio(ids, force=args.force)
        log("SMOKE TEST 完成")
        return 0

    keys = resolve_seasons(args.season)
    log(f"預計處理 {len(keys)} 季: {', '.join(keys)}")

    # 逐季處理並只留下路徑。不 concat 全部：
    # 12 季合起來約 850 萬球 x 88 欄，進記憶體要 7 GB 以上。
    season_paths: list[Path] = []
    for k in keys:
        try:
            pth = fetch_statcast_season(k, args.force)
        except Exception as exc:  # noqa: BLE001
            log(f"! {k} 整季失敗: {exc}")
            continue
        if pth is None:
            continue
        season_paths.append(pth)
        if args.report:
            sanity_report(pd.read_parquet(pth))

    if not season_paths:
        log("! 沒有任何賽季成功，結束")
        return 1

    log(f"--- 共 {len(season_paths)} 季就緒 ---")
    for pth in season_paths:
        log(f"  {pth.name:<28} {pth.stat().st_size / 1e6:>7.1f} MB")

    if not args.skip_bio:
        ids = collect_player_ids(season_paths)
        log(f"需要 {len(ids)} 位球員的身高體重")
        fetch_player_bio(ids, force=args.force)

    log("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
