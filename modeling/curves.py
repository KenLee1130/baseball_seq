"""
curves.py
=========
各種情況的 ROC 曲線與指標表格。

問題 (每個問題是一個二元判斷，在自己的母體上評估)：
  A. 七種事件 (one-vs-rest)
  B. 三段式條件機率：揮棒、判好球|沒揮棒、觸球|揮棒、打進場|觸球、強擊|打進場
  C. 組合指標：CSW、好球、打進場、追打、好球帶內揮棒

分層：全部、球種族、球數、投打慣用手、本打席第幾球、好球帶內外、球季、新人/老將

門檻：在 val 上取每個問題 F1 最大的門檻，存成 thresholds_val.json；
      評估 test 時沿用 val 的門檻，不在 test 上調整。

輸出 runs/<run>/curves_<split>/：
  metrics.csv          每一列 = 問題 x 分組，含 AUC、PR-AUC、Precision、Recall、F1、TP/FP/FN/TN ...
  summary.md           各問題「全部」那一列的摘要
  roc_overview.png     所有問題的整體 ROC
  roc_<問題>.png       單一問題，各分層維度各一格，同維度的分組疊在一起

用法:
    python -m modeling.curves --run runs/<run> --split val
    python -m modeling.curves --run runs/<run> --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_preparation"))

import splits as sp                                   # noqa: E402
from modeling import features as F                    # noqa: E402
from modeling.evaluate import INPUTS, load_model, predict  # noqa: E402
from modeling.outcomes import EVENTS                  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

E = {e: i for i, e in enumerate(EVENTS)}
SWING = ["whiff", "foul_tip", "foul", "in_play_soft", "in_play_hard"]
CONTACT = ["foul_tip", "foul", "in_play_soft", "in_play_hard"]
IN_PLAY = ["in_play_soft", "in_play_hard"]
TAKE = ["ball", "called_strike"]
MIN_GROUP_N = 200       # 分組樣本少於這個數就不計算 AUC 也不畫
EPS = 1e-9


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def psum(p: np.ndarray, events: list[str]) -> np.ndarray:
    return p[:, [E[e] for e in events]].sum(axis=1)


# ---------------------------------------------------------------------------
# 問題定義
# ---------------------------------------------------------------------------

@dataclass
class Question:
    key: str
    label: str
    group: str
    population: Callable[[np.ndarray, np.ndarray], np.ndarray]   # (事件名稱陣列, zone) -> 母體遮罩
    positive: list[str]
    prob: Callable[[np.ndarray], np.ndarray]


ALL = lambda ev, zone: np.ones(len(ev), bool)
IN = lambda events: (lambda ev, zone: np.isin(ev, events))
EVENT_ZH = {"ball": "壞球", "called_strike": "看好球", "whiff": "揮空", "foul_tip": "擦棒被捕",
            "foul": "界外", "in_play_soft": "弱擊", "in_play_hard": "強擊"}

QUESTIONS: list[Question] = (
    [Question(e, f"{EVENT_ZH[e]}", "A 七種事件", ALL, [e], (lambda i: lambda p: p[:, i])(E[e])) for e in EVENTS]
    + [
        Question("swing", "揮棒", "B 三段式", ALL, SWING, lambda p: psum(p, SWING)),
        Question("called_strike_given_take", "判好球 | 沒揮棒", "B 三段式", IN(TAKE), ["called_strike"],
                 lambda p: p[:, E["called_strike"]] / (psum(p, TAKE) + EPS)),
        Question("contact_given_swing", "觸球 | 揮棒", "B 三段式", IN(SWING), CONTACT,
                 lambda p: psum(p, CONTACT) / (psum(p, SWING) + EPS)),
        Question("in_play_given_contact", "打進場 | 觸球", "B 三段式", IN(CONTACT), IN_PLAY,
                 lambda p: psum(p, IN_PLAY) / (psum(p, CONTACT) + EPS)),
        Question("hard_given_in_play", "強擊 | 打進場", "B 三段式", IN(IN_PLAY), ["in_play_hard"],
                 lambda p: p[:, E["in_play_hard"]] / (psum(p, IN_PLAY) + EPS)),
        Question("csw", "CSW (看好球 + 揮空)", "C 組合指標", ALL, ["called_strike", "whiff"],
                 lambda p: psum(p, ["called_strike", "whiff"])),
        Question("strike", "好球 (壞球以外)", "C 組合指標", ALL, [e for e in EVENTS if e != "ball"],
                 lambda p: 1 - p[:, E["ball"]]),
        Question("in_play", "打進場", "C 組合指標", ALL, IN_PLAY, lambda p: psum(p, IN_PLAY)),
        Question("chase", "追打 (好球帶外揮棒)", "C 組合指標", lambda ev, zone: zone >= 11, SWING,
                 lambda p: psum(p, SWING)),
        Question("zone_swing", "好球帶內揮棒", "C 組合指標", lambda ev, zone: (zone >= 1) & (zone <= 9), SWING,
                 lambda p: psum(p, SWING)),
    ]
)


# ---------------------------------------------------------------------------
# 分層定義
# ---------------------------------------------------------------------------

def count_group(count: str) -> str:
    if count == "0-0":
        return "第一球 0-0"
    if count == "3-2":
        return "滿球數 3-2"
    b, s = map(int, count.split("-"))
    if s > b:
        return "投手領先"
    if b > s:
        return "打者領先"
    return "平手 (1-1, 2-2)"


def build_slices(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_pitch = (~d["pad"]).sum(axis=1)
    count = np.array(F.COUNT_VOCAB)[d["ctx_cat"][:, F.CTX_CAT.index("count_state")]]
    zone = d["zone"]
    return {
        "全部": np.full(len(zone), "全部", dtype=object),
        "球種族": np.array(F.FAMILY_VOCAB, dtype=object)[d["family"]],
        "球數": np.array([count_group(c) for c in count], dtype=object),
        "投打慣用手": np.array(F.MATCHUP_VOCAB, dtype=object)[d["ctx_cat"][:, F.CTX_CAT.index("matchup")]],
        "本打席第幾球": np.where(n_pitch >= 7, "第 7 球以後", np.char.add("第 ", np.char.add(n_pitch.astype(str), " 球"))).astype(object),
        "好球帶內外": np.where((zone >= 1) & (zone <= 9), "好球帶內", np.where(zone >= 11, "好球帶外", "未知")).astype(object),
        "球季": d["season"].astype(str).astype(object),
        "打者類型": np.where(d["is_rookie"], "新人 (無上一季)", "老將").astype(object),
    }


# ---------------------------------------------------------------------------
# 資料與預測
# ---------------------------------------------------------------------------

def load_split_with_meta(split: str) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for season in sp.SPLITS[split]:
        arr = F.load_cached(season, mmap=True)
        keep = np.asarray(arr["m_event"])
        for k in INPUTS + ["y_event", "meta_zone", "meta_is_rookie"]:
            parts.setdefault(k, []).append(np.asarray(arr[k])[keep])
        parts.setdefault("season", []).append(np.full(int(keep.sum()), season))
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["zone"], d["is_rookie"] = d.pop("meta_zone"), d.pop("meta_is_rookie")
    d["family"] = d["token_cat"][:, -1, 0]
    return d


def predictions(run_dir: Path, split: str, device: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    d = load_split_with_meta(split)
    log(f"{split} {sp.SPLITS[split]}: {len(d['y_event']):,} 樣本，推論中...")
    p = predict(load_model(run_dir, dev), d, dev)
    return p, d


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------

def f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + EPS)
    return float(thr[int(np.argmax(f1))])


def row_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    n, pos = len(y), int(y.sum())
    out = {"n": n, "positives": pos, "actual_rate": pos / n if n else np.nan, "mean_pred": float(s.mean()) if n else np.nan}
    if n >= MIN_GROUP_N and 0 < pos < n:
        s_clip = np.clip(s, 1e-7, 1 - 1e-7)
        out.update(auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                   pr_auc_lift=average_precision_score(y, s) / (pos / n),
                   brier=brier_score_loss(y, s_clip), log_loss=log_loss(y, s_clip, labels=[0, 1]))
    pred = s >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    out.update(threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn,
               precision=tp / (tp + fp) if tp + fp else np.nan,
               recall=tp / (tp + fn) if tp + fn else np.nan,
               specificity=tn / (tn + fp) if tn + fp else np.nan,
               accuracy=(tp + tn) / n if n else np.nan,
               flagged_rate=(tp + fp) / n if n else np.nan)
    p_, r_ = out["precision"], out["recall"]
    out["f1"] = 2 * p_ * r_ / (p_ + r_) if p_ and r_ and not np.isnan(p_ + r_) else np.nan
    return out


def get_thresholds(run_dir: Path, split: str, p: np.ndarray, d: dict, device: str) -> dict[str, float]:
    path = run_dir / "thresholds_val.json"
    if split != "val":
        if not path.exists():
            log("尚無 val 門檻，先在 val 上計算 (門檻只從 val 取得)")
            pv, dv = predictions(run_dir, "val", device)
            _compute_thresholds(pv, dv, path)
        return json.loads(path.read_text())["thresholds"]
    return _compute_thresholds(p, d, path)


def _compute_thresholds(p: np.ndarray, d: dict, path: Path) -> dict[str, float]:
    ev = np.array(EVENTS)[d["y_event"]]
    thr = {}
    for q in QUESTIONS:
        pop = q.population(ev, d["zone"])
        thr[q.key] = f1_threshold(np.isin(ev[pop], q.positive), q.prob(p)[pop])
    path.write_text(json.dumps({"rule": "val 上 F1 最大", "thresholds": thr}, ensure_ascii=False, indent=2))
    log(f"門檻 -> {path}")
    return thr


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------

def plot_question(q: Question, ev, zone, score, slices, thr, title_suffix, out: Path) -> None:
    pop = q.population(ev, zone)
    y_all, s_all = np.isin(ev[pop], q.positive), score[pop]
    dims = list(slices)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))
    for ax, dim in zip(axes.flat, dims):
        groups = slices[dim][pop]
        order = pd.Series(groups).value_counts().index
        for g in order:
            m = groups == g
            y, s = y_all[m], s_all[m]
            if m.sum() < MIN_GROUP_N or y.all() or not y.any():
                continue
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.6, label=f"{g}  AUC {roc_auc_score(y, s):.3f}  (n={m.sum():,})")
            if dim == "全部":
                pred = s >= thr
                ax.scatter([(pred & ~y).sum() / max((~y).sum(), 1)], [(pred & y).sum() / max(y.sum(), 1)],
                           color="red", zorder=5, label=f"門檻 {thr:.3f} 的操作點")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_title(dim, fontsize=13)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"ROC：{q.label}  ({q.group}，正例比例 {y_all.mean():.1%})  {title_suffix}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_overview(ev, zone, p, title_suffix, out: Path) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(26, 13))
    for ax, q in zip(axes.flat, QUESTIONS):
        pop = q.population(ev, zone)
        y, s = np.isin(ev[pop], q.positive), q.prob(p)[pop]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_title(f"{q.label}\nAUC {roc_auc_score(y, s):.3f}｜正例 {y.mean():.1%}", fontsize=11)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(QUESTIONS):]:
        ax.axis("off")
    fig.suptitle(f"所有問題的整體 ROC  {title_suffix}", fontsize=17)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(run_dir: Path, split: str, device: str = "cuda:0") -> pd.DataFrame:
    run_dir = Path(run_dir)
    out = run_dir / f"curves_{split}"
    out.mkdir(exist_ok=True)
    p, d = predictions(run_dir, split, device)
    thresholds = get_thresholds(run_dir, split, p, d, device)
    ev = np.array(EVENTS)[d["y_event"]]
    slices = build_slices(d)
    suffix = f"｜{run_dir.name}｜{split} {sp.SPLITS[split]}"

    rows = []
    for q in QUESTIONS:
        pop = q.population(ev, d["zone"])
        y_all, s_all = np.isin(ev[pop], q.positive), q.prob(p)[pop]
        for dim, groups in slices.items():
            g_pop = groups[pop]
            for g in pd.Series(g_pop).value_counts().index:
                m = g_pop == g
                rows.append({"group_type": q.group, "question": q.label, "question_key": q.key,
                             "dimension": dim, "group": g, **row_metrics(y_all[m], s_all[m], thresholds[q.key])})
        plot_question(q, ev, d["zone"], q.prob(p), slices, thresholds[q.key], suffix, out / f"roc_{q.key}.png")
        log(f"  {q.label}: 完成")
    plot_overview(ev, d["zone"], p, suffix, out / "roc_overview.png")

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    write_summary(table, out / "summary.md", suffix)
    log(f"-> {out}")
    return table


def write_summary(table: pd.DataFrame, path: Path, suffix: str) -> None:
    t = table[table["dimension"] == "全部"]
    fmt = lambda v, pct=False: "-" if pd.isna(v) else (f"{v:.1%}" if pct else f"{v:.3f}")
    lines = [f"# 各情況指標摘要 {suffix}", "",
             "門檻：val 上 F1 最大 (test 沿用 val 的門檻)。PR-AUC 倍數 = PR-AUC / 正例比例 (隨機猜 = 1)。", "",
             "| 類別 | 問題 | 樣本 | 正例比例 | 平均預測 | AUC | PR-AUC | PR-AUC 倍數 | 門檻 | Precision | Recall | F1 | Specificity | Accuracy |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r.group_type} | {r.question.replace('|', chr(92) + '|')} | {r.n:,} | {fmt(r.actual_rate, True)} | {fmt(r.mean_pred, True)} "
                     f"| {fmt(r.auc)} | {fmt(r.pr_auc)} | {fmt(r.pr_auc_lift)} | {fmt(r.threshold)} | {fmt(r.precision)} "
                     f"| {fmt(r.recall)} | {fmt(r.f1)} | {fmt(r.specificity)} | {fmt(r.accuracy)} |")
    lines += ["", "完整分層 (球種族、球數、投打慣用手、本打席第幾球、好球帶內外、球季、打者類型) 見 metrics.csv 與各 roc_*.png。"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="各情況的 ROC 曲線與指標表格")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=list(sp.SPLITS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(Path(args.run), args.split, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
