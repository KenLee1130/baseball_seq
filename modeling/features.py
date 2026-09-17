"""
features.py
===========
把逐球資料轉成模型輸入張量。**訓練、評估、反事實分析都必須呼叫這一份**，
不得在別處重寫特徵建構 (舊專案曾因反事實腳本自己重建特徵而用錯座標)。

每一個樣本 = 一顆「可當目標」的球，輸入包含：
  token_num  [N, L, F]  最近 L 顆球 (含本球) 的數值特徵，本球在最後一格
  token_cat  [N, L, 2]  球種族、打者反應 (本球的反應被遮蔽)
  pad        [N, L]     True = 補位 (打席的球不足 L 顆)
  ctx_num    [N, G]     情境與打者輪廓的數值特徵
  ctx_cat    [N, K]     球數、壘包出局、左右對戰、前一打席結果
  標籤與遮罩            三段式目標 + 三類擊球，各有自己的母體遮罩

設計原則：
  1. 前後球的差異 (速差、位置差、tunneling) 在這裡由窗口內相鄰的球即時計算，
     而不是讀 preprocess 的 prev1_* 欄位。這樣反事實替換某一顆球後，
     差異特徵會自動跟著更新，不會出現「球換了、速差沒換」的矛盾。
  2. 標準化參數 (Normalizer) 只從 train 估計並存檔；val/test/反事實一律載入同一份。
  3. 類別詞彙寫死在程式裡，不從資料學，確保不同切分的編碼一致。

用法:
    python -m modeling.features --fit           # 用 train 估計標準化參數
    python -m modeling.features --build         # 對所有切分產生快取張量
    python -m modeling.features --fit --build
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preparation"))

import data_spec as spec          # noqa: E402
import splits as sp               # noqa: E402
from batter_profile import FAMILIES, REGIONS  # noqa: E402

SEQ_LEN = 6
PA_KEY = ["game_pk", "at_bat_number"]

# ---------------------------------------------------------------------------
# 詞彙 (0 一律保留給補位)
# ---------------------------------------------------------------------------

PAD = "<pad>"
MASK = "<mask>"
FAMILY_VOCAB = [PAD, *FAMILIES, "other"]
OUTCOME_VOCAB = [PAD, MASK, "ball", "called_strike", "whiff", "foul", "in_play", "hbp", "other"]
COUNT_VOCAB = [PAD] + [f"{b}-{s}" for b in range(4) for s in range(3)]
BASE_OUT_VOCAB = [PAD] + [str(i) for i in range(24)]
PLATOON_VOCAB = [PAD, "same", "opposite"]
PREV_PA_VOCAB = [PAD, "none", "out", "k", "bb_hbp", "single", "xbh"]

CAT_VOCABS = {
    "pitch_family": FAMILY_VOCAB,
    "pitch_outcome": OUTCOME_VOCAB,
    "count_state": COUNT_VOCAB,
    "base_out_state": BASE_OUT_VOCAB,
    "platoon": PLATOON_VOCAB,
    "prev1_pa_result_class": PREV_PA_VOCAB,
}

# ---------------------------------------------------------------------------
# 特徵清單
# ---------------------------------------------------------------------------

# 每顆球自己的物理量 (打者視角)
TOKEN_BASE = ["plate_x_bv", "plate_z_norm", "release_speed", "effective_speed", "speed_vs_own_fastball",
              "pfx_x_bv", "pfx_z", "release_spin_rate", "release_extension", "release_pos_x_bv", "release_pos_z"]
# 與同打席前一顆球的差異 (在窗口內即時計算)
TOKEN_DIFF = ["d_release_speed", "d_plate_x_bv", "d_plate_z_norm", "d_pfx_x_bv", "d_pfx_z",
              "plate_dist", "release_dist", "tunnel_ratio"]
# 只有歷史球才有的結果量；本球為 0
TOKEN_HIST = ["ev_measured"]
TOKEN_FLAGS = ["has_prev", "ev_present"]          # 0/1，不標準化
TOKEN_NUM = TOKEN_BASE + TOKEN_DIFF + TOKEN_HIST + TOKEN_FLAGS
TOKEN_CAT = ["pitch_family", "pitch_outcome"]

CTX_BASE = ["inning", "outs_when_up", "n_runners", "lineup_slot", "nth_pa_vs_pitcher", "pitcher_pitch_count",
            "n_thruorder_pitcher", "bat_score_diff", "pitcher_days_since_prev_game", "inning_batters_faced",
            "prev_pa_reached_count", "age_bat", "age_pit"]
PROFILE_PREV = (
    ["prev_season_swing_rate", "prev_season_whiff_rate", "prev_season_chase_rate", "prev_season_bip_rate",
     "prev_season_hardhit_rate", "prev_season_pull_rate", "prev_season_ev"]
    + [f"prev_season_{m}_{f}" for f in FAMILIES for m in ("chase_rate", "whiff_rate", "hardhit_rate", "ev")]
    + [f"prev_season_{m}_{f}_{r}" for f in FAMILIES for r in REGIONS for m in ("swing_rate", "whiff_rate")]
)
# 揮棒機制：train (2016-2023) 的輪廓來源年份都沒有 bat tracking，全是缺值；
# 若放進模型，val/test 才突然出現數值 -> 分布偏移。預設不使用。
PROFILE_BAT_TRACKING = ["prev_season_mean_bat_speed", "prev_season_fast_swing_rate",
                        "prev_season_mean_swing_length", "prev_season_mean_attack_angle"]
PROFILE_RECENT = ["recent_pull_rate"] + [f"recent_pull_rate_{f}" for f in FAMILIES]
CTX_FLAGS = ["is_rookie", "recent_window_sufficient"]
CTX_CAT = ["count_state", "base_out_state", "platoon", "prev1_pa_result_class"]

LABELS = {
    # 名稱: (欄位, 母體遮罩的產生方式)
    "y_swing": "is_swing",
    "y_contact": "is_contact",
    "y_ev": "ev_measured",
    "y_contact3": "contact3",
}


@dataclass
class FeatureConfig:
    seq_len: int = SEQ_LEN
    use_bat_tracking: bool = False
    use_recent_profile: bool = True

    @property
    def ctx_num(self) -> list[str]:
        cols = CTX_BASE + PROFILE_PREV
        if self.use_bat_tracking:
            cols += PROFILE_BAT_TRACKING
        if self.use_recent_profile:
            cols += PROFILE_RECENT
        return cols + CTX_FLAGS

    def to_dict(self) -> dict:
        return {"seq_len": self.seq_len, "use_bat_tracking": self.use_bat_tracking,
                "use_recent_profile": self.use_recent_profile}


# ---------------------------------------------------------------------------
# 欄位準備
# ---------------------------------------------------------------------------

def input_columns(cfg: FeatureConfig) -> list[str]:
    """從 splits.load_season 讀取時需要的欄位。"""
    raw = ["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "stand", "p_throws",
           "plate_x_bv", "plate_z_norm", "release_speed", "effective_speed", "speed_vs_own_fastball",
           "pfx_x_bv", "pfx_z", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z",
           "pitch_family", "pitch_outcome", "ev_measured", "is_model_target",
           "is_swing", "is_contact", "contact3", "count_state", "base_out_state",
           "prev1_pa_result_class"] + CTX_BASE
    return list(dict.fromkeys(raw))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """補上 features 需要、但 preprocess 沒有直接給的欄位。反事實建構的列也要走這裡。"""
    df = df.copy()
    side = np.where(df["stand"] == "L", -1.0, 1.0)
    df["release_pos_x_bv"] = df["release_pos_x"] * side
    df["platoon"] = np.where(df["stand"] == df["p_throws"], "same", "opposite")
    df["prev1_pa_result_class"] = df["prev1_pa_result_class"].fillna("none")
    df["base_out_state"] = df["base_out_state"].astype("Int64").astype(str).replace("<NA>", PAD)
    df["bat_score_diff"] = df["bat_score_diff"].clip(-10, 10)
    df["inning"] = df["inning"].clip(upper=10)
    df["pitcher_days_since_prev_game"] = df["pitcher_days_since_prev_game"].clip(upper=30)
    return df


def encode_cat(series: pd.Series, vocab: list[str]) -> np.ndarray:
    lookup = {v: i for i, v in enumerate(vocab)}
    other = lookup.get("other", 0)
    return series.map(lambda v: lookup.get(v, other if pd.notna(v) else 0)).to_numpy(np.int16)


# ---------------------------------------------------------------------------
# 標準化 (只從 train 估計)
# ---------------------------------------------------------------------------

@dataclass
class Normalizer:
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    fitted_on: list[int] = field(default_factory=list)

    def transform(self, values: np.ndarray, cols: list[str]) -> np.ndarray:
        """values [..., len(cols)]。缺值標準化後填 0 (= 訓練集平均)。旗標欄位不動。"""
        mean = np.array([self.mean.get(c, 0.0) for c in cols], dtype=np.float32)
        std = np.array([self.std.get(c, 1.0) for c in cols], dtype=np.float32)
        out = (values.astype(np.float32) - mean) / std
        return np.nan_to_num(out, nan=0.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        return cls(**json.loads(path.read_text()))


NORMALIZER_PATH = spec.SPLIT_DIR / "normalizer.json"
FLAG_COLS = set(TOKEN_FLAGS) | set(CTX_FLAGS)


def fit_normalizer(cfg: FeatureConfig, split: str = "train") -> Normalizer:
    """逐季累積平均與變異數，不把 8 季同時載入。只用可當目標的球。"""
    token_cols = TOKEN_BASE + TOKEN_DIFF + TOKEN_HIST
    ctx_cols = [c for c in cfg.ctx_num if c not in FLAG_COLS]
    s1: dict[str, float] = {}
    s2: dict[str, float] = {}
    n: dict[str, float] = {}
    seasons = []
    for df in sp.iter_split(split, input_columns(cfg) + cfg.ctx_num + ["use_as_target"]):
        seasons.append(int(df["season"].iloc[0]))
        df = add_token_diffs(prepare(df))
        t = df[df["use_as_target"]]
        for c in token_cols + ctx_cols:
            v = t[c].to_numpy(np.float64)
            v = v[~np.isnan(v)]
            s1[c] = s1.get(c, 0.0) + v.sum()
            s2[c] = s2.get(c, 0.0) + (v ** 2).sum()
            n[c] = n.get(c, 0.0) + len(v)
        log(f"  fit {seasons[-1]}: {len(t):,} 球")
    mean = {c: s1[c] / n[c] if n[c] else 0.0 for c in s1}
    std = {c: float(np.sqrt(max(s2[c] / n[c] - mean[c] ** 2, 0.0))) if n[c] else 1.0 for c in s1}
    std = {c: (v if v > 1e-6 else 1.0) for c, v in std.items()}
    return Normalizer(mean=mean, std=std, config=cfg.to_dict(), fitted_on=seasons)


# ---------------------------------------------------------------------------
# 張量建構
# ---------------------------------------------------------------------------

def add_token_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """每顆球與同打席前一顆球的差異。df 必須已依打席與 pitch_number 排序。"""
    df = df.copy()
    first = (df.groupby(PA_KEY, sort=False).cumcount() == 0).to_numpy()
    prev = lambda c: np.where(first, np.nan, np.roll(df[c].to_numpy(np.float64), 1))
    for c in ["release_speed", "plate_x_bv", "plate_z_norm", "pfx_x_bv", "pfx_z"]:
        df[f"d_{c}"] = df[c].to_numpy(np.float64) - prev(c)
    df["plate_dist"] = np.hypot(df["d_plate_x_bv"], df["d_plate_z_norm"])
    df["release_dist"] = np.hypot(df["release_pos_x"] - prev("release_pos_x"),
                                  df["release_pos_z"] - prev("release_pos_z"))
    df["tunnel_ratio"] = df["plate_dist"] / np.clip(df["release_dist"], 1 / 12, None)
    df["has_prev"] = (~first).astype(np.float32)
    df["ev_present"] = df["ev_measured"].notna().astype(np.float32)
    return df


def build_arrays(df: pd.DataFrame, norm: Normalizer, cfg: FeatureConfig,
                 target_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """把一季 (或任意依打席排序的一批球) 轉成模型輸入。

    target_mask: 哪些列要產生樣本；預設用 use_as_target 欄位。
    反事實分析傳入自己建構的列，並只對被替換的那顆球產生樣本。
    """
    df = add_token_diffs(prepare(df)).reset_index(drop=True)
    if target_mask is None:
        target_mask = df["use_as_target"].to_numpy(bool)
    L = cfg.seq_len
    pos = df.groupby(PA_KEY, sort=False).cumcount().to_numpy()
    targets = np.flatnonzero(target_mask)

    # 窗口索引：最後一格是目標球本身，往前取同打席的球
    offsets = np.arange(L - 1, -1, -1)
    idx = targets[:, None] - offsets[None, :]
    valid = pos[targets][:, None] >= offsets[None, :]
    idx = np.where(valid, idx, 0)

    # -- 數值 token --
    scaled = np.concatenate([
        norm.transform(df[TOKEN_BASE + TOKEN_DIFF + TOKEN_HIST].to_numpy(np.float64),
                       TOKEN_BASE + TOKEN_DIFF + TOKEN_HIST),
        df[TOKEN_FLAGS].to_numpy(np.float32),
    ], axis=1)
    token_num = np.where(valid[..., None], scaled[idx], 0.0).astype(np.float32)
    # 本球的結果量必須遮蔽 (那是要預測的東西)
    for c in TOKEN_HIST + ["ev_present"]:
        token_num[:, -1, TOKEN_NUM.index(c)] = 0.0

    # -- 類別 token --
    fam = encode_cat(df["pitch_family"], FAMILY_VOCAB)
    out = encode_cat(df["pitch_outcome"], OUTCOME_VOCAB)
    token_cat = np.stack([np.where(valid, fam[idx], 0), np.where(valid, out[idx], 0)], axis=-1).astype(np.int16)
    token_cat[:, -1, 1] = OUTCOME_VOCAB.index(MASK)

    # -- 情境 --
    ctx_cols = cfg.ctx_num
    t = df.iloc[targets]
    ctx_scaled = [c for c in ctx_cols if c not in FLAG_COLS]
    ctx_num = np.concatenate([
        norm.transform(t[ctx_scaled].to_numpy(np.float64), ctx_scaled),
        t[[c for c in ctx_cols if c in FLAG_COLS]].astype(np.float32).to_numpy(),
    ], axis=1)
    ctx_cat = np.stack([encode_cat(t[c], CAT_VOCABS[c]) for c in CTX_CAT], axis=1)

    # -- 標籤與母體遮罩 --
    labels = {
        "y_swing": t["is_swing"].to_numpy(np.float32),
        "y_contact": t["is_contact"].to_numpy(np.float32),
        "m_contact": t["is_swing"].to_numpy(bool),                 # Stage 2 母體：有揮棒
        "y_ev": t["ev_measured"].fillna(0.0).to_numpy(np.float32),
        "m_ev": t["ev_measured"].notna().to_numpy(bool),           # Stage 3 母體：量得到初速
        "y_contact3": t["contact3"].clip(lower=0).to_numpy(np.int64),
        "m_contact3": (t["contact3"] >= 0).to_numpy(bool),         # 打進場缺初速者排除
    }
    meta = {k: t[k].to_numpy() for k in ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"]}

    return {"token_num": token_num, "token_cat": token_cat, "pad": ~valid,
            "ctx_num": ctx_num.astype(np.float32), "ctx_cat": ctx_cat, **labels,
            **{f"meta_{k}": v for k, v in meta.items()}}


def feature_names(cfg: FeatureConfig) -> dict[str, list[str]]:
    scaled_ctx = [c for c in cfg.ctx_num if c not in FLAG_COLS]
    return {"token_num": TOKEN_NUM, "token_cat": TOKEN_CAT,
            "ctx_num": scaled_ctx + [c for c in cfg.ctx_num if c in FLAG_COLS], "ctx_cat": CTX_CAT}


# ---------------------------------------------------------------------------
# 快取
# ---------------------------------------------------------------------------

ARRAY_DIR = spec.SPLIT_DIR / "arrays"


def build_cache(cfg: FeatureConfig, norm: Normalizer) -> None:
    """每季一個資料夾，每個張量一個 .npy (可用 mmap 讀取)。"""
    for name, seasons in sp.SPLITS.items():
        for season in seasons:
            t0 = time.time()
            df = sp.load_season(season, input_columns(cfg) + cfg.ctx_num)
            arrays = build_arrays(df, norm, cfg)
            out = ARRAY_DIR / str(season)
            out.mkdir(parents=True, exist_ok=True)
            for k, v in arrays.items():
                np.save(out / f"{k}.npy", v)
            (out / "info.json").write_text(json.dumps({
                "season": season, "split": name, "n_samples": int(len(arrays["ctx_num"])),
                "feature_config": cfg.to_dict(), "feature_names": feature_names(cfg),
                "shapes": {k: list(v.shape) for k, v in arrays.items()},
            }, ensure_ascii=False, indent=2))
            size = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e6
            log(f"  {season} [{name}] {len(arrays['ctx_num']):,} 樣本, {size:.0f} MB, {time.time() - t0:.0f}s")


def load_cached(season: int, mmap: bool = True) -> dict[str, np.ndarray]:
    folder = ARRAY_DIR / str(season)
    return {f.stem: np.load(f, mmap_mode="r" if mmap else None) for f in folder.glob("*.npy")}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="建立模型輸入張量")
    ap.add_argument("--fit", action="store_true", help="用 train 估計標準化參數")
    ap.add_argument("--build", action="store_true", help="產生所有切分的快取張量")
    ap.add_argument("--use-bat-tracking", action="store_true", help="納入揮棒機制特徵 (見 PROFILE_BAT_TRACKING 的說明)")
    args = ap.parse_args()
    cfg = FeatureConfig(use_bat_tracking=args.use_bat_tracking)

    if args.fit:
        log("估計標準化參數 (train)...")
        norm = fit_normalizer(cfg)
        norm.save(NORMALIZER_PATH)
        log(f"-> {NORMALIZER_PATH}")
    if args.build:
        norm = Normalizer.load(NORMALIZER_PATH)
        if norm.config != cfg.to_dict():
            raise SystemExit(f"標準化參數的設定 {norm.config} 與目前 {cfg.to_dict()} 不同，請先 --fit")
        log("建立快取張量...")
        build_cache(cfg, norm)
    if not (args.fit or args.build):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
