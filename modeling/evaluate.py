"""
evaluate.py
===========
評估一次訓練的結果。訓練結束會自動在 val 上跑一次；test 只在模型定案後手動跑。

輸出 runs/<run>/eval_<split>/：
  metrics.json   所有數字
  report.md      人看的摘要

評估內容：
  1. 整體      7 類 log loss，和「只用類別比例猜」的基準比較
  2. 各事件    one-vs-rest AUC、PR-AUC、平均預測 vs 實際比例
  3. 三段式    由 7 類機率換算的條件機率，各自在自己的母體上評估
               揮棒 / 判好球|沒揮棒 / 觸球|揮棒 / 打進場|觸球 / 強擊|打進場 / 強擊 (所有球)
  4. 分層      強擊與揮棒，依球種族、球數、投打慣用手組合、本打席第幾球分組
  5. 校準      強擊與揮棒的十分位校準表
  6. 歷史消融  把前面的球換成別的打席的，看 AUC 掉多少、預測變多少 (模型有沒有用到序列)

用法:
    python -m modeling.evaluate --run runs/event7_base_xxx --split val
    python -m modeling.evaluate --run runs/event7_base_xxx --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_preparation"))

import splits as sp                                      # noqa: E402
from modeling import features as F                       # noqa: E402
from modeling.model import ModelConfig, PitchSequenceModel  # noqa: E402
from modeling.outcomes import EVENTS                     # noqa: E402

E = {e: i for i, e in enumerate(EVENTS)}
SWING = ["whiff", "foul_tip", "foul", "in_play_soft", "in_play_hard"]
CONTACT = ["foul_tip", "foul", "in_play_soft", "in_play_hard"]
IN_PLAY = ["in_play_soft", "in_play_hard"]
TAKE = ["ball", "called_strike"]
INPUTS = ["token_num", "token_cat", "pad", "ctx_num", "ctx_cat"]


def _print(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------------

def load_model(run_dir: Path, device: torch.device) -> PitchSequenceModel:
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    model = PitchSequenceModel(ckpt["n_token_num"], ckpt["n_ctx_num"], ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_split(split: str) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for season in sp.SPLITS[split]:
        arr = F.load_cached(season, mmap=True)
        keep = np.asarray(arr["m_event"])
        for k in INPUTS + ["y_event"]:
            parts.setdefault(k, []).append(np.asarray(arr[k])[keep])
    return {k: np.concatenate(v) for k, v in parts.items()}


@torch.no_grad()
def predict(model, data: dict[str, np.ndarray], device: torch.device, batch_size: int = 16384) -> np.ndarray:
    out = []
    n = len(data["y_event"])
    for s in range(0, n, batch_size):
        x = [torch.from_numpy(np.ascontiguousarray(data[k][s:s + batch_size])).to(device) for k in INPUTS]
        x[0], x[1], x[4] = x[0].float(), x[1].long(), x[4].long()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(*x)
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------

def psum(p: np.ndarray, events: list[str]) -> np.ndarray:
    return p[:, [E[e] for e in events]].sum(axis=1)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(int)
    out = {"n": int(len(y)), "actual": float(y.mean()), "mean_pred": float(p.mean())}
    if 0 < y.sum() < len(y):
        out.update(auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                   brier=float(brier_score_loss(y, p)))
    return out


def stage_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """由 7 類機率換算三段式條件機率，並在各自的母體上評估。"""
    ev = np.array(EVENTS)[y]
    eps = 1e-9
    stages = {
        "swing": (np.ones(len(y), bool), psum(p, SWING), np.isin(ev, SWING)),
        "called_strike|take": (np.isin(ev, TAKE), p[:, E["called_strike"]] / (psum(p, TAKE) + eps),
                               ev == "called_strike"),
        "contact|swing": (np.isin(ev, SWING), psum(p, CONTACT) / (psum(p, SWING) + eps), np.isin(ev, CONTACT)),
        "in_play|contact": (np.isin(ev, CONTACT), psum(p, IN_PLAY) / (psum(p, CONTACT) + eps), np.isin(ev, IN_PLAY)),
        "hard|in_play": (np.isin(ev, IN_PLAY), p[:, E["in_play_hard"]] / (psum(p, IN_PLAY) + eps),
                         ev == "in_play_hard"),
        "hard (所有球)": (np.ones(len(y), bool), p[:, E["in_play_hard"]], ev == "in_play_hard"),
    }
    return {name: binary_metrics(target[pop], prob[pop]) for name, (pop, prob, target) in stages.items()}


def stratified(p_bin: np.ndarray, y_bin: np.ndarray, groups: pd.Series, min_n: int = 500) -> list[dict]:
    rows = []
    for g, idx in groups.groupby(groups).indices.items():
        if len(idx) < min_n:
            continue
        rows.append({"group": str(g), **binary_metrics(y_bin[idx], p_bin[idx])})
    return sorted(rows, key=lambda r: -r["n"])


def calibration(p_bin: np.ndarray, y_bin: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.unique(np.quantile(p_bin, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, p_bin, side="right") - 1, 0, len(edges) - 2)
    return [{"bin": int(b), "mean_pred": float(p_bin[bins == b].mean()), "actual": float(y_bin[bins == b].mean()),
             "n": int((bins == b).sum())} for b in range(len(edges) - 1) if (bins == b).any()]


def shuffle_history(data: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """把每個樣本「本球以前的球」換成另一個隨機樣本的。本球的差異特徵也跟著換 (那來自前一球)。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data["y_event"]))
    out = dict(data)
    for k in ["token_num", "token_cat", "pad"]:
        x = np.array(data[k])
        x[:, :-1] = x[perm, :-1]
        out[k] = x
    prev_dependent = [F.TOKEN_NUM.index(c) for c in F.TOKEN_DIFF + ["has_prev"]]
    out["token_num"][:, -1, prev_dependent] = np.asarray(data["token_num"])[perm][:, -1, prev_dependent]
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def evaluate_run(run_dir: Path, split: str, log=_print, device: str = "cuda:0") -> dict:
    run_dir = Path(run_dir)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, dev)
    data = load_split(split)
    y = data["y_event"]
    log(f"評估 {run_dir.name} on {split} ({sp.SPLITS[split]}): {len(y):,} 樣本")

    p = predict(model, data, dev)
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    log_loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())

    metrics = {
        "run": run_dir.name, "split": split, "seasons": sp.SPLITS[split], "n": int(len(y)),
        "log_loss": log_loss,
        "per_event": {e: binary_metrics(y == i, p[:, i]) for i, e in enumerate(EVENTS)},
        "stages": stage_metrics(p, y),
    }

    family = pd.Series(np.array(F.FAMILY_VOCAB)[data["token_cat"][:, -1, 0]])
    count = pd.Series(np.array(F.COUNT_VOCAB)[data["ctx_cat"][:, F.CTX_CAT.index("count_state")]])
    matchup = pd.Series(np.array(F.MATCHUP_VOCAB)[data["ctx_cat"][:, F.CTX_CAT.index("matchup")]])
    nth = pd.Series(np.minimum((~np.asarray(data["pad"])).sum(1), 7)).map(lambda k: f"第 {k} 球" if k < 7 else "第 7 球以後")
    hard_p, hard_y = p[:, E["in_play_hard"]], (y == E["in_play_hard"])
    swing_p, swing_y = psum(p, SWING), np.isin(np.array(EVENTS)[y], SWING)
    metrics["stratified"] = {
        "hard_by_family": stratified(hard_p, hard_y, family),
        "hard_by_count": stratified(hard_p, hard_y, count),
        "swing_by_family": stratified(swing_p, swing_y, family),
        "swing_by_count": stratified(swing_p, swing_y, count),
        "hard_by_matchup": stratified(hard_p, hard_y, matchup),
        "swing_by_matchup": stratified(swing_p, swing_y, matchup),
        "hard_by_pitch_index": stratified(hard_p, hard_y, nth),
        "swing_by_pitch_index": stratified(swing_p, swing_y, nth),
    }
    metrics["calibration"] = {"hard": calibration(hard_p, hard_y), "swing": calibration(swing_p, swing_y)}

    p_shuf = predict(model, shuffle_history(data), dev)
    ablation = {}
    for name, (pop, prob, target) in {
        "swing": (slice(None), lambda q: psum(q, SWING), swing_y),
        "contact|swing": (swing_y, lambda q: psum(q, CONTACT) / (psum(q, SWING) + 1e-9),
                          np.isin(np.array(EVENTS)[y], CONTACT)),
        "hard (所有球)": (slice(None), lambda q: q[:, E["in_play_hard"]], hard_y),
    }.items():
        a, b = prob(p)[pop], prob(p_shuf)[pop]
        ablation[name] = {"auc": float(roc_auc_score(target[pop], a)),
                          "auc_shuffled": float(roc_auc_score(target[pop], b)),
                          "mean_abs_change": float(np.abs(a - b).mean())}
    metrics["history_ablation"] = ablation

    out = run_dir / f"eval_{split}"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    report = render_report(metrics, ckpt.get("baseline_val_loss"))
    (out / "report.md").write_text(report)
    log("\n" + report)
    return metrics


def render_report(m: dict, baseline: float | None) -> str:
    pct = lambda v: f"{v * 100:.1f}%"
    f3 = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    lines = [f"# 評估：{m['run']} / {m['split']} {m['seasons']}", "",
             f"樣本 {m['n']:,}｜log loss **{m['log_loss']:.4f}**"
             + (f"（只用類別比例的基準 {baseline:.4f}，降低 {(1 - m['log_loss'] / baseline):.2%}）" if baseline else ""),
             "", "## 三段式 (由 7 類機率換算)", "",
             "| 條件機率 | 母體 | 實際 | 平均預測 | AUC | PR-AUC |", "|---|---|---|---|---|---|"]
    for name, s in m["stages"].items():
        lines.append(f"| {name} | {s['n']:,} | {pct(s['actual'])} | {pct(s['mean_pred'])} | {f3(s.get('auc'))} | {f3(s.get('pr_auc'))} |")
    lines += ["", "## 各事件", "", "| 事件 | 實際 | 平均預測 | AUC | PR-AUC |", "|---|---|---|---|---|"]
    for e, s in m["per_event"].items():
        lines.append(f"| {e} | {pct(s['actual'])} | {pct(s['mean_pred'])} | {f3(s.get('auc'))} | {f3(s.get('pr_auc'))} |")
    for key, title in [("hard_by_family", "強擊 (所有球) × 球種族"), ("hard_by_count", "強擊 (所有球) × 球數"),
                       ("hard_by_matchup", "強擊 (所有球) × 投打慣用手"), ("swing_by_pitch_index", "揮棒 × 本打席第幾球")]:
        lines += ["", f"## {title}", "", "| 分組 | 樣本 | 實際 | 平均預測 | AUC |", "|---|---|---|---|---|"]
        for r in m["stratified"][key]:
            lines.append(f"| {r['group']} | {r['n']:,} | {pct(r['actual'])} | {pct(r['mean_pred'])} | {f3(r.get('auc'))} |")
    lines += ["", "## 歷史消融 (把前面的球換成別的打席的)", "",
              "| 目標 | AUC | 打亂後 AUC | 預測平均變動 |", "|---|---|---|---|"]
    for name, a in m["history_ablation"].items():
        lines.append(f"| {name} | {a['auc']:.4f} | {a['auc_shuffled']:.4f} | {a['mean_abs_change'] * 100:.2f} 個百分點 |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="評估一次訓練")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=list(sp.SPLITS))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    evaluate_run(Path(args.run), args.split, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
