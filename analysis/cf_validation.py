"""
cf_validation.py
================
反事實驗證：在資料裡找「情境幾乎相同、只有下一球不同」的例子，
比較模型預測的差距和實際發生的差距是否一致。

情境 (situation) 的定義：
  投手類型   依球速、位移、球種使用率分群 (KMeans，投手-球季為單位)
  投手慣用手 + 打者站位
  球數
  打者類型   上一季揮空率 x 強擊率，各分高 / 中 / 低 (9 類，新人排除)
  前一球     球種族 + 位置 + 打者反應 (打席第一球則為「無」)

下一球 (variant)：球種族 x 位置 (好球帶上 / 中 / 下、帶外高 / 低 / 內角 / 外角)

兩種比較：
  1. 實際投出的球：同一情境中，下一球 A 與 B 的實際發生率差距 vs 模型對這些球的平均預測差距
  2. 反事實交換：把投 A 的球換成「該投手自己實際投過的一顆 B」(隨機抽一顆真實的球，
     不用平均值，避免平均出一個不存在的位置)，
     用 features.build_arrays 重建輸入 (速差等差異特徵會跟著更新) 讓模型預測，
     預測差距 = 交換後的平均預測 - 原本 A 的平均預測，再和實際差距比較

注意：這仍是觀察資料。投手選 A 或 B 可能有我們看不到的原因 (當天球況、捕手、戰術)，
所以這是在檢驗模型的條件機率預測，不能證明因果效果。

用法:
    python -m analysis.cf_validation --run runs/<run>
輸出 runs/<run>/cf_validation/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_preparation"))

import splits as sp                                    # noqa: E402
from batter_profile import FAMILIES                    # noqa: E402
from modeling import features as F                     # noqa: E402
from modeling.evaluate import load_model, predict      # noqa: E402
from modeling.outcomes import EVENTS                   # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEASONS = [2024, 2025, 2026]
N_PITCHER_TYPES = 6
MIN_PITCHER_PITCHES = 300      # 投手-球季至少投這麼多球才分群
MIN_CELL_N = 50                # 一個「情境 x 下一球」至少幾球才比較
MAX_SWAP_PER_CELL = 400        # 每格最多拿幾球做反事實交換 (控制運算量)
MIN_PITCHER_VARIANT_N = 10     # 投手投過某種下一球至少幾次，才用他的平均物理量做交換
SEED = 42

EVENT_ZH = {"ball": "壞球", "called_strike": "看好球", "whiff": "揮空", "foul_tip": "擦棒被捕",
            "foul": "界外", "in_play_soft": "弱擊", "in_play_hard": "強擊"}
OUTCOMES = {  # 名稱: 由哪些事件組成
    **{EVENT_ZH[e]: [e] for e in EVENTS},
    "揮棒": ["whiff", "foul_tip", "foul", "in_play_soft", "in_play_hard"],
    "CSW": ["called_strike", "whiff"],
    "好球 (壞球以外)": [e for e in EVENTS if e != "ball"],
    "打進場": ["in_play_soft", "in_play_hard"],
}
FAMILY_ZH = {"fastball": "四縫線", "sinker": "伸卡", "cutter": "卡特", "slider": "滑球",
             "curveball": "曲球", "changeup": "變速"}
IN_ZONE_ROW = {1: "上", 2: "上", 3: "上", 4: "中", 5: "中", 6: "中", 7: "下", 8: "下", 9: "下"}


def region(df: pd.DataFrame) -> pd.Series:
    """好球帶內：Savant zone 的上 / 中 / 下三排。
    好球帶外：以座標分成高、低、內角、外角四個緊密方向。
    不用 Savant 的 11-14 區：那四區各自橫跨左右兩側，同一區的球位置差很多。"""
    inside = df["zone"].map(IN_ZONE_ROW)
    out = np.select([df["plate_z_norm"] > 1.0, df["plate_z_norm"] < 0.0, df["plate_x_bv"] < 0],
                    ["帶外高", "帶外低", "帶外內角"], "帶外外角")
    r = inside.fillna(pd.Series(out, index=df.index))
    return r.where(df["zone"].notna(), "未知")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. 資料
# ---------------------------------------------------------------------------

def load_data(cfg: F.FeatureConfig) -> pd.DataFrame:
    extra = ["p_throws", "pfx_x", "arm_angle", "prev_season_whiff_rate", "prev_season_hardhit_rate", "is_rookie"]
    parts = []
    for season in SEASONS:
        df = sp.load_season(season, F.input_columns(cfg) + cfg.ctx_num + extra)
        parts.append(df)
        log(f"  讀取 {season}: {len(df):,} 球")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)
    df["region"] = region(df)
    df["variant"] = df["pitch_family"].map(FAMILY_ZH).fillna("其他") + "_" + df["region"]
    g = df.groupby(["game_pk", "at_bat_number"], sort=False)
    prev_var, prev_out = g["variant"].shift(1), g["pitch_outcome"].shift(1)
    df["prev1"] = np.where(prev_var.isna(), "無 (打席第一球)", prev_var.fillna("") + "/" + prev_out.fillna(""))
    df["event"] = pd.Series(pd.NA, index=df.index, dtype=object)
    return df


def pitcher_types(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """投手-球季分群。特徵以慣用手翻正 (水平位移正 = 手臂側)。"""
    t = df[df["use_as_target"]].copy()
    t["season"] = t["season"].astype(int)
    t["arm_side_pfx"] = np.where(t["p_throws"] == "L", -t["pfx_x"], t["pfx_x"])
    key = ["pitcher", "season"]
    n = t.groupby(key).size().rename("n")
    fb = t[t["pitch_family"].isin(["fastball", "sinker"])].groupby(key).agg(
        fb_speed=("release_speed", "mean"), fb_arm_side=("arm_side_pfx", "mean"), fb_ride=("pfx_z", "mean"))
    usage = (t.groupby(key)["pitch_family"].value_counts(normalize=True).unstack(fill_value=0.0)
               .reindex(columns=list(FAMILIES), fill_value=0.0).add_prefix("use_"))
    misc = t.groupby(key).agg(extension=("release_extension", "mean"), release_z=("release_pos_z", "mean"),
                              arm_angle=("arm_angle", "mean"))
    feat = pd.concat([n, fb, usage, misc], axis=1)
    feat = feat[feat["n"] >= MIN_PITCHER_PITCHES].dropna()

    cols = [c for c in feat.columns if c != "n"]
    km = KMeans(N_PITCHER_TYPES, n_init=20, random_state=SEED).fit(StandardScaler().fit_transform(feat[cols]))
    feat["pitcher_type"] = km.labels_

    prof = feat.groupby("pitcher_type")[cols].mean()
    names = {}
    for k, r in prof.iterrows():
        top = r[[f"use_{f}" for f in FAMILIES]].sort_values(ascending=False).index[:2]
        mix = "+".join(f"{FAMILY_ZH[c[4:]]}{r[c]:.0%}" for c in top)
        names[k] = f"T{k} {r['fb_speed']:.1f}mph {mix}"
    feat["pitcher_type_name"] = feat["pitcher_type"].map(names)
    prof.insert(0, "name", pd.Series(names))
    prof.insert(1, "n_pitcher_seasons", feat.groupby("pitcher_type").size())
    return feat.reset_index(), prof


def batter_types(df: pd.DataFrame) -> pd.Series:
    t = df["use_as_target"] & ~df["is_rookie"].astype(bool)
    q = lambda c: df.loc[t, c].quantile([1 / 3, 2 / 3]).to_numpy()
    w, h = q("prev_season_whiff_rate"), q("prev_season_hardhit_rate")
    lab = lambda v, e: np.select([v <= e[0], v <= e[1]], ["低", "中"], "高")
    out = pd.Series("揮空" + lab(df["prev_season_whiff_rate"], w) + "_強擊" + lab(df["prev_season_hardhit_rate"], h),
                    index=df.index)
    return out.where(~df["is_rookie"].astype(bool), "新人")


# ---------------------------------------------------------------------------
# 2. 模型預測
# ---------------------------------------------------------------------------

def predict_rows(model, df: pd.DataFrame, mask: np.ndarray, cfg, norm, device) -> np.ndarray:
    arr = F.build_arrays(df, norm, cfg, target_mask=mask)
    return predict(model, arr, device)


def event_index(df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    from modeling.outcomes import event_label
    lab = event_label(df.loc[rows, "description"], df.loc[rows, "launch_speed"])
    return lab.map({e: i for i, e in enumerate(EVENTS)}).to_numpy()


def outcome_matrix(p_or_onehot: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({name: p_or_onehot[:, [EVENTS.index(e) for e in evs]].sum(axis=1)
                         for name, evs in OUTCOMES.items()})


SWAP_COLS = ["release_speed", "effective_speed", "plate_x_bv", "plate_z_norm", "pfx_x_bv", "pfx_z",
             "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z"]


def counterfactual_swap(model, df, swaps: pd.DataFrame, cfg, norm, device):
    """swaps: 每列 = (原始列號 row, 用來替換的真實球列號 donor)。

    把原始列所在打席「第 1 球到這一球」複製成一個新的虛擬打席，
    最後一球的物理量與位置換成 donor 那顆球 (同一位投手投的 B)，再交給 features.build_arrays
    (差異特徵、序列都由 features.py 重新計算)。
    """
    pa_pos = df.groupby(["game_pk", "at_bat_number"], sort=False).cumcount().to_numpy()
    ends = swaps["row"].to_numpy()
    lengths = pa_pos[ends] + 1
    starts = ends - pa_pos[ends]
    idx = np.repeat(starts, lengths) + (np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths))
    synth = df.iloc[idx].reset_index(drop=True)
    synth["game_pk"] = -1
    synth["at_bat_number"] = np.repeat(np.arange(len(swaps)), lengths)

    last = np.cumsum(lengths) - 1
    donor = df.iloc[swaps["donor"].to_numpy()]
    own_fb = synth.loc[last, "release_speed"].to_numpy() - synth.loc[last, "speed_vs_own_fastball"].to_numpy()
    for c in SWAP_COLS + ["pitch_family", "zone"]:
        synth.iloc[last, synth.columns.get_loc(c)] = donor[c].to_numpy()
    synth.iloc[last, synth.columns.get_loc("speed_vs_own_fastball")] = donor["release_speed"].to_numpy() - own_fb

    mask = np.zeros(len(synth), bool)
    mask[last] = True
    return predict_rows(model, synth, mask, cfg, norm, device)


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def run(run_dir: Path, device: str) -> None:
    run_dir = Path(run_dir)
    out = run_dir / "cf_validation"
    out.mkdir(exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg = F.FeatureConfig(**run_cfg["features"])
    norm = F.Normalizer.load(run_dir / "normalizer.json")
    model = load_model(run_dir, dev)
    rng = np.random.default_rng(SEED)

    log("1. 讀取 2024-2026 資料")
    df = load_data(cfg)

    log("2. 投手分群")
    ptype, prof = pitcher_types(df)
    prof.round(3).to_csv(out / "pitcher_types.csv", encoding="utf-8-sig")
    log("\n" + prof[["name", "n_pitcher_seasons", "fb_speed", "fb_arm_side", "fb_ride", "arm_angle"]].round(2).to_string())
    df = df.merge(ptype[["pitcher", "season", "pitcher_type_name"]], on=["pitcher", "season"], how="left")
    df["batter_type"] = batter_types(df)

    log("3. 模型預測所有目標球")
    target = df["use_as_target"].to_numpy() & df["pitcher_type_name"].notna().to_numpy()
    ev = event_index(df, np.flatnonzero(target))
    keep = ~pd.isna(ev)
    rows = np.flatnonzero(target)[keep]
    mask = np.zeros(len(df), bool)
    mask[rows] = True
    probs = predict_rows(model, df, mask, cfg, norm, dev)
    onehot = np.eye(len(EVENTS))[ev[keep].astype(int)]

    situation = ["pitcher_type_name", "p_throws", "stand", "count_state", "batter_type", "prev1"]
    base = df.loc[rows, ["pitcher"] + situation + ["variant", "region"]].copy()
    base["row"] = rows
    obs_rows = outcome_matrix(onehot).set_index(rows)
    pred_rows = outcome_matrix(probs).set_index(rows)
    base = base[(base["batter_type"] != "新人") & (base["region"] != "未知") & ~base["variant"].str.startswith("其他")]
    log(f"   可用樣本 {len(base):,} 球")

    cell_size = base.groupby(situation + ["variant"]).size()
    ok_cells = cell_size[cell_size >= MIN_CELL_N].reset_index()[situation + ["variant"]]
    per_sit = ok_cells.groupby(situation).size()
    cells = ok_cells.merge(per_sit[per_sit >= 2].rename("k").reset_index(), on=situation)
    base = base.merge(cells[situation + ["variant"]], on=situation + ["variant"])
    log(f"   可比較的情境 {cells[situation].drop_duplicates().shape[0]:,} 個，情境 x 下一球 {len(cells):,} 格，"
        f"涵蓋 {len(base):,} 球")

    keys = [base[c] for c in situation + ["variant"]]
    O = obs_rows.loc[base["row"]].reset_index(drop=True)
    P = pred_rows.loc[base["row"]].reset_index(drop=True)
    cell_obs = O.groupby(keys).mean()
    cell_pred = P.groupby(keys).mean()
    cell_size = base.groupby(situation + ["variant"]).size()

    log("4. 反事實交換：把 A 換成同一位投手實際投過的 B")
    # 每位投手每種下一球的真實球池 (至少 MIN_PITCHER_VARIANT_N 顆)
    pool = pd.DataFrame({"pitcher": df.loc[rows, "pitcher"].to_numpy(), "variant": df.loc[rows, "variant"].to_numpy(),
                         "row": rows})
    pool = pool[~pool["variant"].str.endswith("未知")].sort_values(["pitcher", "variant"], kind="stable")
    sizes = pool.groupby(["pitcher", "variant"], sort=True).size()
    sizes = sizes[sizes >= MIN_PITCHER_VARIANT_N]
    pool = (pool.merge(sizes.rename("pool_n").reset_index(), on=["pitcher", "variant"])
                .sort_values(["pitcher", "variant"], kind="stable").reset_index(drop=True))
    pool_rows = pool["row"].to_numpy()
    pool_start = pool.groupby(["pitcher", "variant"], sort=True).cumcount().rsub(np.arange(len(pool))).to_numpy()
    first_of = pd.Series(pool_start, index=pd.MultiIndex.from_frame(pool[["pitcher", "variant"]])).groupby(level=[0, 1]).first()

    parts = []
    variants_by_sit = cells.groupby(situation)["variant"].apply(list)
    # 每格隨機取最多 MAX_SWAP_PER_CELL 球：先整體打亂，再取每格的前 N 球
    shuffled = base.sample(frac=1.0, random_state=SEED)
    sampled = shuffled[shuffled.groupby(situation + ["variant"]).cumcount() < MAX_SWAP_PER_CELL]
    for sit, sub in sampled.groupby(situation):
        for b in variants_by_sit.loc[sit]:
            a_rows = sub[sub["variant"] != b]
            parts.append(a_rows[["row", "pitcher", "variant"] + situation].assign(to_variant=b))
    swaps = pd.concat(parts, ignore_index=True).rename(columns={"variant": "from_variant"})
    key = pd.MultiIndex.from_arrays([swaps["pitcher"], swaps["to_variant"]])
    pos = first_of.index.get_indexer(key)
    swaps = swaps[pos >= 0].reset_index(drop=True)
    pos = pos[pos >= 0]
    n_pool = sizes.reindex(first_of.index).to_numpy()[pos]
    offset = (rng.random(len(swaps)) * n_pool).astype(int)
    swaps["donor"] = pool_rows[first_of.to_numpy()[pos] + offset]
    log(f"   交換樣本 {len(swaps):,} 筆")

    swap_pred = []
    step = 100_000
    for s in range(0, len(swaps), step):
        swap_pred.append(counterfactual_swap(model, df, swaps.iloc[s:s + step], cfg, norm, dev))
    swap_pred = outcome_matrix(np.concatenate(swap_pred))
    orig_pred = pred_rows.loc[swaps["row"]].reset_index(drop=True)
    cf = pd.concat([swaps[situation + ["from_variant", "to_variant"]], (swap_pred - orig_pred).add_prefix("cfΔ_")], axis=1)
    cf_mean = cf.groupby(situation + ["from_variant", "to_variant"]).mean()
    cf_n = cf.groupby(situation + ["from_variant", "to_variant"]).size().rename("n_swapped")

    log("5. 兩兩比較")
    cell_obs_f, cell_pred_f = cell_obs.reset_index(), cell_pred.reset_index()
    size_f = cell_size.rename("n").reset_index()
    pairs = cf_mean.join(cf_n).reset_index().rename(columns={"from_variant": "A", "to_variant": "B"})
    for tag, col in [("A", "A"), ("B", "B")]:
        o = cell_obs_f.rename(columns={"variant": col, **{k: f"obs_{k}_{tag}" for k in OUTCOMES}})
        p_ = cell_pred_f.rename(columns={"variant": col, **{k: f"pred_{k}_{tag}" for k in OUTCOMES}})
        n_ = size_f.rename(columns={"variant": col, "n": f"n_{tag}"})
        pairs = pairs.merge(o, on=situation + [col]).merge(p_, on=situation + [col]).merge(n_, on=situation + [col])
    for o in OUTCOMES:
        na, nb = pairs["n_A"], pairs["n_B"]
        pa, pb = pairs[f"obs_{o}_A"], pairs[f"obs_{o}_B"]
        # 標準誤用平滑後的比例 (加 1 成功 / 1 失敗)，避免 0% 的格子標準誤 = 0 而權重爆炸
        sa, sb = (pa * na + 1) / (na + 2), (pb * nb + 1) / (nb + 2)
        pairs[f"{o}|實際Δ"] = pb - pa
        pairs[f"{o}|實際Δ_se"] = np.sqrt(sa * (1 - sa) / na + sb * (1 - sb) / nb)
        pairs[f"{o}|預測Δ_實際投球"] = pairs[f"pred_{o}_B"] - pairs[f"pred_{o}_A"]
        pairs[f"{o}|預測Δ_反事實"] = pairs[f"cfΔ_{o}"]
    pairs = pairs[pairs["A"] < pairs["B"]].reset_index(drop=True)  # A->B 與 B->A 只留一個方向
    pairs.to_csv(out / "pairs.csv", index=False, encoding="utf-8-sig", float_format="%.4f")

    summary = summarize(pairs)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    plot_scatter(pairs, out / "scatter_counterfactual.png", "預測Δ_反事實", run_dir.name)
    plot_scatter(pairs, out / "scatter_factual.png", "預測Δ_實際投球", run_dir.name)
    write_report(out, prof, summary, pairs, cells, situation, run_dir.name)
    log(f"-> {out}")


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for o in OUTCOMES:
        d = pairs[f"{o}|實際Δ"].to_numpy()
        se = pairs[f"{o}|實際Δ_se"].to_numpy()
        w = 1 / se ** 2
        sig = np.abs(d) > 2 * se
        for kind in ["預測Δ_實際投球", "預測Δ_反事實"]:
            p = pairs[f"{o}|{kind}"].to_numpy()
            mw = lambda x: np.sum(w * x) / np.sum(w)
            cov = mw((p - mw(p)) * (d - mw(d)))
            corr = cov / np.sqrt(mw((p - mw(p)) ** 2) * mw((d - mw(d)) ** 2))
            slope = cov / mw((p - mw(p)) ** 2)
            rows.append({
                "結果": o, "比較方式": "實際投出的球" if kind == "預測Δ_實際投球" else "反事實交換",
                "配對數": len(d), "加權相關係數": corr, "迴歸斜率 (理想=1)": slope,
                "方向一致 (實際差距顯著的配對)": np.mean(np.sign(p[sig]) == np.sign(d[sig])) if sig.any() else np.nan,
                "實際差距顯著的配對數": int(sig.sum()),
                "預測落在實際 95% 區間內": np.mean(np.abs(p - d) <= 2 * se),
                "平均|實際Δ|": np.mean(np.abs(d)), "平均|預測Δ|": np.mean(np.abs(p)),
            })
    return pd.DataFrame(rows)


def plot_scatter(pairs: pd.DataFrame, path: Path, kind: str, run_name: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    for ax, o in zip(axes.flat, OUTCOMES):
        p, d, se = pairs[f"{o}|{kind}"], pairs[f"{o}|實際Δ"], pairs[f"{o}|實際Δ_se"]
        size = np.clip((pairs["n_A"] + pairs["n_B"]) / 20, 5, 200)
        ax.errorbar(p, d, yerr=2 * se, fmt="none", ecolor="lightgray", elinewidth=0.6, zorder=1)
        ax.scatter(p, d, s=size, alpha=0.5, zorder=2)
        lim = np.nanmax(np.abs(np.r_[p, d])) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "--", color="red", lw=1, label="y = x")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{o}  (相關 {np.corrcoef(p, d)[0, 1]:.2f})", fontsize=13)
        ax.set_xlabel("模型預測的差距 (B - A)")
        ax.set_ylabel("實際差距 (B - A)")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    title = "反事實交換" if kind == "預測Δ_反事實" else "實際投出的球"
    fig.suptitle(f"下一球從 A 換成 B：模型預測差距 vs 實際差距 ({title})｜{run_name}｜2024-2026｜誤差線 = 實際差距 ±2SE",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_report(out, prof, summary, pairs, cells, situation, run_name) -> None:
    f3 = lambda v: "-" if pd.isna(v) else f"{v:.3f}"
    pct = lambda v: "-" if pd.isna(v) else f"{v:.1%}"
    lines = [f"# 反事實驗證｜{run_name}｜2024-2026", "",
             f"情境 = 投手類型 + 投手慣用手 + 打者站位 + 球數 + 打者類型 + 前一球；每格至少 {MIN_CELL_N} 球。",
             f"可比較的情境 {cells[situation].drop_duplicates().shape[0]:,} 個、配對 {len(pairs):,} 組。", "",
             "## 投手類型", "", "| 類型 | 投手-球季數 | 速球均速 | 速球手臂側位移 | 速球垂直位移 | 手臂角度 |", "|---|---|---|---|---|---|"]
    for _, r in prof.iterrows():
        lines.append(f"| {r['name']} | {int(r['n_pitcher_seasons'])} | {r['fb_speed']:.1f} | {r['fb_arm_side']:.2f} | {r['fb_ride']:.2f} | {r['arm_angle']:.1f} |")
    for kind in ["實際投出的球", "反事實交換"]:
        s = summary[summary["比較方式"] == kind]
        lines += ["", f"## {kind}", "",
                  "| 結果 | 配對數 | 加權相關 | 斜率 (理想=1) | 方向一致 (顯著配對) | 顯著配對數 | 預測落在實際 95% 區間 | 平均\\|實際Δ\\| | 平均\\|預測Δ\\| |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in s.itertuples(index=False):
            lines.append(f"| {r[0]} | {r[2]:,} | {f3(r[3])} | {f3(r[4])} | {pct(r[5])} | {r[6]:,} | {pct(r[7])} | {pct(r[8])} | {pct(r[9])} |")
    lines += ["", "- **加權相關**：以實際差距的變異數倒數加權，樣本多的配對權重較大",
              "- **斜率**：實際差距對預測差距迴歸；< 1 表示模型高估差距、> 1 表示低估",
              "- **方向一致**：只看實際差距超過 2 個標準誤的配對，模型預測的方向是否相同",
              "- 圖：`scatter_counterfactual.png`、`scatter_factual.png`；逐組數字：`pairs.csv`"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="反事實驗證")
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(Path(args.run), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
