"""
outcomes.py
===========
把模型的三段式預測，對應到棒球場上每一球實際會發生的事件、球數變化與打席結局。

三段式 (data_spec.TARGETS) 只回答三個條件機率：
  s = P(揮棒)
  c = P(觸球 | 揮棒)
  EV | 觸球

但場上的事件需要再多三個條件機率，否則無法決定球數怎麼走、打席有沒有結束：
  k = P(判好球 | 沒揮棒)        沒揮棒時是好球還是壞球
  f = P(打進場 | 觸球)          觸球時是界外 (打席繼續) 還是打進場 (打席結束)
  q = P(擦棒被捕 | 界外類觸球)   兩好球時擦棒被接住就是三振
  h = P(初速 > 95 | 打進場)      強擊只對打進場的球有意義

事件樹：

  投球
  ├─ 沒揮棒 (1-s)
  │   ├─ 壞球          (1-s)(1-k)       b+1；三壞球時 -> 保送
  │   └─ 好球          (1-s) k          s+1；兩好球時 -> 三振
  └─ 揮棒 s
      ├─ 揮空          s(1-c)           s+1；兩好球時 -> 三振
      └─ 觸球 s c
          ├─ 擦棒被捕  s c (1-f) q      s+1；兩好球時 -> 三振
          ├─ 界外      s c (1-f)(1-q)   兩好球以下 s+1；兩好球時球數不變
          └─ 打進場 s c f               打席結束
              ├─ 強擊  s c f h
              └─ 弱擊  s c f (1-h)

用法:
    python -m modeling.outcomes --fit     # 用 train 估計每個 (球數, 事件) 的平均得分價值
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preparation"))

import data_spec as spec  # noqa: E402
import splits as sp       # noqa: E402

HARD_HIT_MPH = 95.0
EVENTS = ["ball", "called_strike", "whiff", "foul_tip", "foul", "in_play_soft", "in_play_hard"]
COUNTS = [(b, s) for b in range(4) for s in range(3)]
TERMINALS = ["strikeout", "walk", "in_play_soft", "in_play_hard"]
RUN_VALUE_PATH = spec.SPLIT_DIR / "run_values.json"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. 條件機率 -> 事件機率
# ---------------------------------------------------------------------------

@dataclass
class StageProbs:
    """一顆球的條件機率。欄位可以是純量或同形狀的 numpy 陣列 (批次)。"""
    swing: np.ndarray                 # s
    called_strike_given_take: np.ndarray  # k
    contact_given_swing: np.ndarray   # c
    in_play_given_contact: np.ndarray  # f
    foul_tip_given_foul: np.ndarray   # q
    hard_given_in_play: np.ndarray    # h


def event_probs(p: StageProbs) -> dict[str, np.ndarray]:
    s, k, c = p.swing, p.called_strike_given_take, p.contact_given_swing
    f, q, h = p.in_play_given_contact, p.foul_tip_given_foul, p.hard_given_in_play
    return {
        "ball": (1 - s) * (1 - k),
        "called_strike": (1 - s) * k,
        "whiff": s * (1 - c),
        "foul_tip": s * c * (1 - f) * q,
        "foul": s * c * (1 - f) * (1 - q),
        "in_play_soft": s * c * f * (1 - h),
        "in_play_hard": s * c * f * h,
    }


def event_label(description: pd.Series, launch_speed: pd.Series) -> pd.Series:
    """把 Statcast description 對應到 EVENTS。

    回傳 NaN 的情況 (不在事件樹內，不可當標籤)：
      - 觸擊、觸身球
      - 打進場但沒有初速：無法判斷強弱，不可默默當成弱擊
    """
    bip = description == "hit_into_play"
    return pd.Series(np.select(
        [description.isin(["ball", "blocked_ball"]),
         description == "called_strike",
         description.isin(["swinging_strike", "swinging_strike_blocked"]),
         description == "foul_tip",
         description == "foul",
         bip & (launch_speed > HARD_HIT_MPH),
         bip & launch_speed.notna()],
        ["ball", "called_strike", "whiff", "foul_tip", "foul", "in_play_hard", "in_play_soft"],
        default=None,
    ), index=description.index)


# ---------------------------------------------------------------------------
# 2. 球數轉移
# ---------------------------------------------------------------------------

def transition(balls: int, strikes: int, event: str) -> tuple[int, int] | str:
    """回傳下一個球數 (b, s)，或打席結局字串。"""
    if event == "ball":
        return "walk" if balls == 3 else (balls + 1, strikes)
    if event in ("called_strike", "whiff", "foul_tip"):
        return "strikeout" if strikes == 2 else (balls, strikes + 1)
    if event == "foul":
        return (balls, min(strikes + 1, 2))
    if event in ("in_play_soft", "in_play_hard"):
        return event
    raise ValueError(f"未知事件 {event}")


def step(count_dist: dict[tuple[int, int], float], probs_at: dict[tuple[int, int], dict[str, float]]
         ) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    """把「目前球數的機率分布」往前推一球。

    count_dist: {(b, s): 機率}
    probs_at:   每個球數下這一球的事件機率 (反事實時，不同球數可以選不同的球)
    回傳 (下一球的球數分布, 這一球就結束打席的機率)
    """
    nxt: dict[tuple[int, int], float] = {}
    ended = {t: 0.0 for t in TERMINALS}
    for count, w in count_dist.items():
        if w == 0:
            continue
        for ev, p in probs_at[count].items():
            res = transition(*count, ev)
            if isinstance(res, str):
                ended[res] += w * p
            else:
                nxt[res] = nxt.get(res, 0.0) + w * p
    return nxt, ended


# ---------------------------------------------------------------------------
# 3. 得分價值 (只用 train 估計)
# ---------------------------------------------------------------------------

def fit_run_values() -> dict:
    """每個 (球數, 事件) 的平均 delta_run_exp (進攻方視角；對投手越負越好)。"""
    cols = ["description", "launch_speed", "balls", "strikes", "delta_run_exp", "is_model_target"]
    parts = []
    for df in sp.iter_split("train", cols):
        t = df[df["use_as_target"]].copy()
        t["event"] = event_label(t["description"], t["launch_speed"])
        parts.append(t.dropna(subset=["event", "delta_run_exp"])[["balls", "strikes", "event", "delta_run_exp"]])
        log(f"  run value {int(df['season'].iloc[0])}: {len(parts[-1]):,} 球")
    t = pd.concat(parts)
    table = t.groupby(["balls", "strikes", "event"])["delta_run_exp"].agg(["mean", "count"]).reset_index()
    return {
        "fitted_on": sp.SPLITS["train"],
        "note": "delta_run_exp 為進攻方視角，數值越負對投手越有利",
        "values": {f"{int(r.balls)}-{int(r.strikes)}|{r.event}": {"mean": round(float(r["mean"]), 4), "n": int(r["count"])}
                   for _, r in table.iterrows()},
    }


def expected_run_value(probs: dict[str, np.ndarray], balls: int, strikes: int, run_values: dict) -> np.ndarray:
    """一顆球的期望得分價值 = sum_事件 P(事件) x 該球數下該事件的平均價值。"""
    total = 0.0
    for ev, p in probs.items():
        total = total + p * run_values["values"][f"{balls}-{strikes}|{ev}"]["mean"]
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="三段式預測 -> 場上事件")
    ap.add_argument("--fit", action="store_true", help="用 train 估計 (球數, 事件) 的得分價值表")
    args = ap.parse_args()
    if not args.fit:
        ap.print_help()
        return 0
    rv = fit_run_values()
    RUN_VALUE_PATH.write_text(json.dumps(rv, ensure_ascii=False, indent=2))
    log(f"-> {RUN_VALUE_PATH} ({len(rv['values'])} 格)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
