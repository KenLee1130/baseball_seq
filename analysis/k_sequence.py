"""
k_sequence.py
=============
「最多投 5 顆球，怎麼配球三振機率最高？」

指定一位投手、一位打者、一個比賽情境，從打席第一球 (0-0) 開始展開一棵樹：
  節點 = 目前的球數與本打席已投的球 (含打者反應)
  選擇 = 下一球投哪一種 (這位投手實際投過的「球種 x 位置」，用一顆代表性的真實球)
  分支 = 模型預測的打者反應；三振、保送、打進場時打席結束
  5 顆球內沒有結束的打席視為沒有三振

比較的策略：
  最佳應變策略   每一球依上一球的結果決定 (樹狀搜尋)
  最佳固定序列   事先決定 5 顆球的順序，不管打者反應
  貪婪策略       每一球只挑「這一球 + 球數推進後的歷史三振率」最高的，不往後看
  投手實際傾向   依投手 2026 年在各球數的實際配球比例隨機投 (模擬)
  隨機配球       依投手 2026 年整體使用比例隨機投 (模擬)
  實際結果       真實打席 (不是模型)，分兩列：
                   對這位打者 (2024-2026，樣本通常很少)
                   對其他同慣用手打者 (2026，排除這位打者)

門檻：投手 2026 年實際三振率 (對同一慣用手打者) 的 1.5 倍，回報每個策略最少幾球達到。

搜尋方式：前 2 球完整展開；第 3 球起每個節點只展開一球評分最高的前 BEAM_K 個候選，
並略過發生機率低於 MIN_PATH_PROB 的路徑、以及剩下球數已不可能三振的節點。
被略過的部分其三振機率以 0 計，所以最佳策略的數字是下界。

用法:
    python -m analysis.k_sequence --run runs/<run>
輸出 runs/<run>/k_sequence/
"""

from __future__ import annotations

import argparse
import json
import re
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_preparation"))

import data_spec as spec                               # noqa: E402
import splits as sp                                    # noqa: E402
from analysis.cf_validation import FAMILY_ZH, SWAP_COLS, region  # noqa: E402
from modeling import features as F                     # noqa: E402
from modeling.evaluate import load_model, predict      # noqa: E402
from modeling.outcomes import EVENTS                   # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PITCHER = ("山本由伸", 808967)
BATTERS = [("Aaron Judge", 592450), ("Kyle Schwarber", 656941), ("Juan Soto", 665742),
           ("Luis Arraez", 650333), ("Bobby Witt Jr.", 677951)]
SEASON = 2026
PROFILE_DATE = "2026-07-01"     # 打者近期輪廓取這天之前的最後一筆
BUDGET = 5
MIN_COMBO_N = 20                # 「球種 x 位置」至少投過幾次才列為候選
BEAM_FROM_DEPTH = 3
BEAM_K = 4
MIN_PATH_PROB = 1e-3
FIXED_BEAM = 40
MC_N = 10_000
THRESHOLD_MULT = 1.5
SEED = 42
CONTEXT = {  # 隨意挑的比賽情境：3 局上、無人出局、壘上無人、平手、打線第一輪、上一棒出局
    "inning": 3, "outs_when_up": 0, "n_runners": 0, "base_out_state": 0, "bat_score_diff": 0,
    "lineup_slot": 3, "nth_pa_vs_pitcher": 1, "n_thruorder_pitcher": 1, "inning_batters_faced": 0,
    "prev_pa_reached_count": 0, "prev1_pa_result_class": "out", "pitcher_pitch_count": 35,
}
CONTEXT_TEXT = "3 局上、無人出局、壘上無人、0:0 平手、打線第一輪、上一棒出局、投手本場已投 34 球"

E = {e: i for i, e in enumerate(EVENTS)}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. 模板：投手、打者、候選球
# ---------------------------------------------------------------------------

class Setup:
    def __init__(self, run_dir: Path, device: str):
        self.run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.cfg = F.FeatureConfig(**self.run_cfg["features"])
        self.norm = F.Normalizer.load(run_dir / "normalizer.json")
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = load_model(run_dir, self.dev)
        self.cols = list(dict.fromkeys(F.input_columns(self.cfg) + self.cfg.ctx_num + ["p_throws", "pfx_x"]))
        log(f"讀取 {SEASON} 資料")
        self.df = sp.load_season(SEASON, self.cols)
        self.bio = pd.read_parquet(spec.RAW_DIR / "player_bio.parquet").set_index("player_id")
        self.cache: dict = {}
        self.n_evals = 0

    def pitcher_rows(self) -> pd.DataFrame:
        p = self.df[(self.df["pitcher"] == PITCHER[1]) & self.df["use_as_target"]].copy()
        if p.empty:
            raise SystemExit(f"投手 {PITCHER} 在 {SEASON} 年沒有資料")
        p["region"] = region(p)
        p["variant"] = p["pitch_family"].map(FAMILY_ZH).fillna("其他") + "_" + p["region"]
        return p[~p["variant"].str.startswith("其他") & (p["region"] != "未知")]

    def batter_template(self, batter_id: int, p_throws: str) -> dict:
        rows = self.df[self.df["batter"] == batter_id]
        if rows.empty:
            raise SystemExit(f"打者 {batter_id} 在 {SEASON} 年沒有資料")
        same = rows[rows["p_throws"] == p_throws]
        stand = (same if len(same) else rows)["stand"].mode().iloc[0]   # 左右開弓者：取面對這個慣用手投手時的站位
        before = rows[rows["game_date"] < PROFILE_DATE]
        row = (before if len(before) else rows).iloc[-1]
        keep = [c for c in self.cfg.ctx_num if c.startswith(("prev_season_", "recent_"))] + \
               ["is_rookie", "recent_window_sufficient", "age_bat"]
        return {"batter": batter_id, "stand": stand, **{c: row[c] for c in keep}}

    def pitcher_template(self, p: pd.DataFrame) -> dict:
        own_fb = float((p["release_speed"] - p["speed_vs_own_fastball"]).median())
        return {"pitcher": PITCHER[1], "p_throws": p["p_throws"].mode().iloc[0], "age_pit": float(p["age_pit"].median()),
                "pitcher_days_since_prev_game": float(p["pitcher_days_since_prev_game"].median()),
                "own_fb": own_fb, "game_date": PROFILE_DATE}

    def candidates(self, p: pd.DataFrame, stand: str) -> pd.DataFrame:
        """同一慣用手打者的球，依「球種 x 位置」分組，每組取最接近組中心的真實球 (medoid)。"""
        q = p[p["stand"] == stand]
        z = (q[SWAP_COLS] - q[SWAP_COLS].mean()) / q[SWAP_COLS].std()
        rows = []
        for v, g in q.groupby("variant"):
            if len(g) < MIN_COMBO_N:
                continue
            zg = z.loc[g.index]
            med = g.loc[((zg - zg.mean()) ** 2).sum(axis=1).idxmin()]
            rows.append({"variant": v, "n": len(g), **{c: med[c] for c in SWAP_COLS + ["pitch_family", "zone"]}})
        cand = pd.DataFrame(rows).reset_index(drop=True)
        total = cand["n"].sum()
        cand["usage"] = cand["n"] / total
        by_count = (q[q["variant"].isin(cand["variant"])].groupby(["count_state", "variant"]).size()
                      .unstack(fill_value=0).reindex(columns=cand["variant"], fill_value=0))
        self.usage_by_count = {c: (r / r.sum()).to_numpy() if r.sum() >= 30 else cand["usage"].to_numpy()
                               for c, r in by_count.iterrows()}
        return cand

    # ---- 模型評估：(歷史, 候選) -> 7 類機率 -------------------------------

    def evaluate(self, items: list[tuple[tuple, int]], cand: pd.DataFrame, bt: dict, pt: dict) -> np.ndarray:
        """items: [(歷史, 候選編號)]；歷史 = ((候選, 反應, 壞球數, 好球數), ...)。有快取。"""
        out = np.zeros((len(items), len(EVENTS)))
        todo = [i for i, it in enumerate(items) if it not in self.cache]
        for s in range(0, len(todo), 50_000):
            chunk = [items[i] for i in todo[s:s + 50_000]]
            probs = self._predict(chunk, cand, bt, pt)
            for it, pr in zip(chunk, probs):
                self.cache[it] = pr
            self.n_evals += len(chunk)
        for i, it in enumerate(items):
            out[i] = self.cache[it]
        return out

    def _predict(self, chunk, cand, bt, pt) -> np.ndarray:
        cand_ids, toks, balls, strikes, sample, pos = [], [], [], [], [], []
        for k, (hist, c) in enumerate(chunk):
            for j, (hc, tok, b, s) in enumerate(hist):
                cand_ids.append(hc); toks.append(tok); balls.append(b); strikes.append(s); sample.append(k); pos.append(j)
            b, s = (0, 0) if not hist else _next_count(hist[-1])
            cand_ids.append(c); toks.append("ball"); balls.append(b); strikes.append(s); sample.append(k); pos.append(len(hist))
        cand_ids, pos = np.array(cand_ids), np.array(pos)
        df = cand.iloc[cand_ids][SWAP_COLS + ["pitch_family", "zone"]].reset_index(drop=True)
        df["speed_vs_own_fastball"] = df["release_speed"] - pt["own_fb"]
        const = {**{k: v for k, v in pt.items() if k != "own_fb"}, **bt, **CONTEXT}
        for k, v in const.items():
            df[k] = v
        df["pitcher_pitch_count"] = CONTEXT["pitcher_pitch_count"] + pos
        df["game_pk"] = -1
        df["at_bat_number"] = sample
        df["pitch_number"] = pos + 1
        df["pitch_outcome"] = toks
        df["count_state"] = [f"{b}-{s}" for b, s in zip(balls, strikes)]
        for c, v in {"ev_measured": np.nan, "launch_speed": np.nan, "description": "ball", "contact3": 0,
                     "is_swing": False, "is_contact": False}.items():
            df[c] = v
        last = np.r_[pos[1:] == 0, True]
        arr = F.build_arrays(df, self.norm, self.cfg, target_mask=last)
        return predict(self.model, arr, self.dev)


def _next_count(step) -> tuple[int, int]:
    """歷史最後一球 (候選, 反應, b, s) 之後的球數。"""
    _, tok, b, s = step
    if tok == "ball":
        return b + 1, s
    if tok in ("called_strike", "whiff"):
        return b, s + 1
    return b, min(s + 1, 2)  # foul


def branches(p: np.ndarray, b: int, s: int) -> tuple[dict, list]:
    """一顆球的結果：打席結束的機率 + 繼續的分支 [(反應, 機率, 新壞球數, 新好球數)]。"""
    end = {"K": 0.0, "BB": 0.0, "soft": p[E["in_play_soft"]], "hard": p[E["in_play_hard"]]}
    kids = []
    if b == 3:
        end["BB"] += p[E["ball"]]
    else:
        kids.append(("ball", p[E["ball"]], b + 1, s))
    if s == 2:
        end["K"] += p[E["called_strike"]] + p[E["whiff"]] + p[E["foul_tip"]]
        kids.append(("foul", p[E["foul"]], b, 2))
    else:
        kids.append(("called_strike", p[E["called_strike"]], b, s + 1))
        kids.append(("whiff", p[E["whiff"]], b, s + 1))
        kids.append(("foul", p[E["foul"]] + p[E["foul_tip"]], b, s + 1))  # 未滿兩好球時擦棒 = 界外
    return end, kids


ZH_TOK = {"ball": "壞球", "called_strike": "看好球", "whiff": "揮空", "foul": "界外"}


def k_possible(s: int, pitches_left: int) -> bool:
    return 3 - s <= pitches_left


# ---------------------------------------------------------------------------
# 2. 策略
# ---------------------------------------------------------------------------

def k_hist_table() -> dict[str, float]:
    """歷史上從各球數出發最後三振的比例 (train 最後一季)，當作搜尋時的評分參考。"""
    cols = ["game_pk", "at_bat_number", "pitch_number", "count_state", "events", "pa_valid"]
    d = pd.read_parquet(spec.INTERIM_DIR / f"pitches_{max(sp.SPLITS['train'])}.parquet", columns=cols)
    d = d[d["pa_valid"]].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    last = d.groupby(["game_pk", "at_bat_number"])["events"].last().fillna("").str.startswith("strikeout")
    d = d.merge(last.rename("k").reset_index(), on=["game_pk", "at_bat_number"])
    return d.groupby("count_state")["k"].mean().to_dict()


def score(p, b, s, khist):
    end, kids = branches(p, b, s)
    return end["K"] + sum(pr * khist[f"{nb}-{ns}"] for _, pr, nb, ns in kids)


def adaptive_tree(st: Setup, cand, bt, pt, khist):
    C = len(cand)
    level = [((), 0, 0, 1.0)]   # (歷史, b, s, 路徑機率)
    nodes = {}                 # 歷史 -> (b, s, 各候選機率, 展開的候選)
    for depth in range(1, BUDGET + 1):
        left = BUDGET - depth + 1
        level = [n for n in level if k_possible(n[2], left)]
        if not level:
            break
        items = [(h, c) for h, _, _, _ in level for c in range(C)]
        probs = st.evaluate(items, cand, bt, pt).reshape(len(level), C, -1)
        nxt = []
        for (h, b, s, pp), pr in zip(level, probs):
            if depth < BEAM_FROM_DEPTH:
                expand = list(range(C))
            else:
                expand = list(np.argsort([-score(pr[c], b, s, khist) for c in range(C)])[:BEAM_K])
            nodes[h] = (b, s, pr, expand)
            if depth == BUDGET:
                continue
            for c in expand:
                _, kids = branches(pr[c], b, s)
                for tok, q, nb, ns in kids:
                    if pp * q >= MIN_PATH_PROB:
                        nxt.append((h + ((c, tok, b, s),), nb, ns, pp * q))
        level = nxt
        log(f"    深度 {depth}: 節點 {len(nodes):,}，模型評估累計 {st.n_evals:,}")
    return nodes


def solve(nodes, budget: int, choose: str = "max", khist=None):
    """在已展開的樹上求策略。回傳 (根節點的結果, 每個節點選的候選)。

    choose="max" 為最佳應變策略；choose="greedy" 為貪婪策略 (以 score 選，不看子樹)。
    結果包含 K, BB, soft, hard, unfinished, pitches，以及 K_within[n]。
    """
    policy = {}

    def rec(h, depth):
        if h not in nodes or depth > budget:
            return None
        b, s, pr, expand = nodes[h]
        best, best_c = None, None
        options = expand if choose == "max" else [int(np.argmax([score(pr[c], b, s, khist) for c in range(len(pr))]))]
        for c in options:
            end, kids = branches(pr[c], b, s)
            res = {"K": end["K"], "BB": end["BB"], "soft": end["soft"], "hard": end["hard"], "unfinished": 0.0,
                   "pitches": 1.0, "K_within": np.zeros(BUDGET + 1)}
            res["K_within"][depth:] += end["K"]
            for tok, q, nb, ns in kids:
                child = rec(h + ((c, tok, b, s),), depth + 1)
                if child is None:
                    res["unfinished"] += q
                    continue
                for k in ("K", "BB", "soft", "hard", "unfinished"):
                    res[k] += q * child[k]
                res["pitches"] += q * child["pitches"]
                res["K_within"] += q * child["K_within"]
            if best is None or res["K"] > best["K"]:
                best, best_c = res, c
        policy[h] = best_c
        return best

    return rec((), 1), policy


def fixed_sequence(st: Setup, cand, bt, pt, khist):
    """最佳固定序列：beam search，每步保留期望分數最高的前 FIXED_BEAM 條。"""
    C = len(cand)
    beams = [((), [((), 0, 0, 1.0)], {"K": 0.0, "BB": 0.0, "soft": 0.0, "hard": 0.0}, [])]
    for depth in range(1, BUDGET + 1):
        items = [(h, c) for _, states, _, _ in beams for h, _, _, _ in states for c in range(C)]
        st.evaluate(items, cand, bt, pt)
        new = []
        for seq, states, acc, curve in beams:
            for c in range(C):
                acc2 = dict(acc)
                nxt = []
                for h, b, s, pp in states:
                    pr = st.cache[(h, c)]
                    end, kids = branches(pr, b, s)
                    for k in acc2:
                        acc2[k] += pp * end[k]
                    nxt += [(h + ((c, tok, b, s),), nb, ns, pp * q) for tok, q, nb, ns in kids]
                sc = acc2["K"] + sum(pp * khist[f"{b}-{s}"] for _, b, s, pp in nxt)
                new.append((seq + (c,), nxt, acc2, curve + [acc2["K"]], sc))
        new.sort(key=lambda x: -(x[4] if depth < BUDGET else x[2]["K"]))
        beams = [x[:4] for x in new[:FIXED_BEAM]]
        log(f"    固定序列 深度 {depth}: 模型評估累計 {st.n_evals:,}")
    ranked = [(seq, acc["K"]) for seq, _, acc, _ in beams[:10]]
    seq, states, acc, curve = beams[0]
    acc["unfinished"] = sum(pp for _, _, _, pp in states)
    return seq, acc, curve, ranked


def simulate(st: Setup, cand, bt, pt, mode: str, rng, khist, policy=None, seq=None):
    """模擬 MC_N 個打席，回傳 5 球內各結果比例與「n 球內三振」曲線。

    mode:
      usage   依投手整體使用比例隨機挑
      count   依投手在該球數的實際比例隨機挑
      fixed   照固定序列 seq
      policy  照樹狀搜尋的策略；搜尋時沒展開的節點改用貪婪選擇
      greedy  每一球挑 score 最高的
    所有策略都用同一套模擬計算結果，數字才能直接比較。
    """
    C = len(cand)
    hist = [()] * MC_N
    bs = [(0, 0)] * MC_N
    steps = [[] for _ in range(MC_N)]
    endings = [""] * MC_N
    alive = np.ones(MC_N, bool)
    ended = {k: np.zeros(MC_N, bool) for k in ("K", "BB", "soft", "hard")}
    k_at = np.full(MC_N, 99)
    for depth in range(1, BUDGET + 1):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        choice = np.full(len(idx), -1)
        for j, i in enumerate(idx):
            key = f"{bs[i][0]}-{bs[i][1]}"
            if mode == "usage":
                choice[j] = rng.choice(C, p=cand["usage"].to_numpy())
            elif mode == "count":
                choice[j] = rng.choice(C, p=st.usage_by_count.get(key, cand["usage"].to_numpy()))
            elif mode == "fixed":
                choice[j] = seq[depth - 1]
            elif mode == "policy" and policy.get(hist[i]) is not None:
                choice[j] = policy[hist[i]]
        need = np.flatnonzero(choice < 0)
        if len(need):
            allp = st.evaluate([(hist[idx[j]], c) for j in need for c in range(C)], cand, bt, pt).reshape(len(need), C, -1)
            for j, pr in zip(need, allp):
                b, s = bs[idx[j]]
                choice[j] = int(np.argmax([score(pr[c], b, s, khist) for c in range(C)]))
        probs = st.evaluate([(hist[i], int(c)) for i, c in zip(idx, choice)], cand, bt, pt)
        for i, c, pr in zip(idx, choice, probs):
            b, s = bs[i]
            ev = EVENTS[rng.choice(len(EVENTS), p=pr / pr.sum())]
            head = f"第 {depth} 球 ({b}-{s}) {cand.at[int(c), 'variant']}"
            if ev in ("in_play_soft", "in_play_hard"):
                ended["soft" if ev == "in_play_soft" else "hard"][i] = True
                alive[i] = False
                steps[i].append(f"{head} → 打進場 → **{'弱擊' if ev == 'in_play_soft' else '強擊'}**")
                endings[i] = "soft" if ev == "in_play_soft" else "hard"
            elif ev == "ball" and b == 3:
                ended["BB"][i], alive[i] = True, False
                steps[i].append(f"{head} → 壞球 → **保送**")
                endings[i] = "BB"
            elif ev in ("called_strike", "whiff", "foul_tip") and s == 2:
                ended["K"][i], alive[i], k_at[i] = True, False, depth
                steps[i].append(f"{head} → 好球 → **三振**")
                endings[i] = "K"
            else:
                tok = {"foul_tip": "foul"}.get(ev, ev)
                hist[i] = hist[i] + ((int(c), tok, b, s),)
                bs[i] = _next_count(hist[i][-1])
                steps[i].append(f"{head} → {ZH_TOK[tok]} → {bs[i][0]}-{bs[i][1]}")
                if depth == BUDGET:
                    endings[i] = "unfinished"
    res = {k: float(v.mean()) for k, v in ended.items()}
    res["unfinished"] = float(alive.mean())
    counts = pd.Series(["；".join(st_) + "|" + e for st_, e in zip(steps, endings)]).value_counts()
    res["paths"] = [(n / MC_N, key.rsplit("|", 1)[0], key.rsplit("|", 1)[1]) for key, n in counts.items()]
    return res, [float((k_at <= n).mean()) for n in range(1, BUDGET + 1)]


def actual_outcomes(stand: str, batter_id: int, who: str, seasons: list[int]):
    """投手的真實打席 (不是模型)。

    who = "batter"  只看對這位打者的打席
          "others"  同一慣用手、但排除這位打者
    回傳 (整體三振率, 5 球內各結果比例, n 球內三振曲線, 打席數)。
    「5 球內」= 打席在第 5 球以內結束；超過 5 球才結束的打席算「5 球後未結束」。
    """
    cols = ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "stand", "events", "description",
            "launch_speed", "pa_valid"]
    d = pd.concat([pd.read_parquet(spec.INTERIM_DIR / f"pitches_{y}.parquet", columns=cols) for y in seasons])
    d = d[(d["pitcher"] == PITCHER[1]) & (d["stand"] == stand) & d["pa_valid"]]
    d = d[d["batter"] == batter_id] if who == "batter" else d[d["batter"] != batter_id]
    d = d.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    pa = d.groupby(["game_pk", "at_bat_number"]).agg(n=("pitch_number", "size"), ev=("events", "last"),
                                                     desc=("description", "last"), ls=("launch_speed", "last"))
    if pa.empty:
        return np.nan, None, [np.nan] * BUDGET, 0
    ev = pa["ev"].fillna("")
    k = ev.str.startswith("strikeout")
    bb = ev.isin(["walk", "intent_walk", "hit_by_pitch"])
    bip = pa["desc"] == "hit_into_play"
    within = pa["n"] <= BUDGET
    res = {"K": float((k & within).mean()), "BB": float((bb & within).mean()),
           "hard": float((bip & (pa["ls"] > 95) & within).mean()),
           "soft": float((bip & ~(pa["ls"] > 95) & within).mean()),
           "unfinished": float((~within).mean())}
    curve = [float((k & (pa["n"] <= n)).mean()) for n in range(1, BUDGET + 1)]
    return float(k.mean()), res, curve, len(pa)


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def describe_policy(nodes, policy, cand, max_depth=2):
    """最佳應變策略的前兩球：第 1 球投什麼；依第 1 球的反應，第 2 球投什麼。"""
    zh = {"ball": "壞球", "called_strike": "看好球", "whiff": "揮空", "foul": "界外"}
    root_c = policy[()]
    lines = [f"- 第 1 球 (投球前 0-0)：**{cand.at[root_c, 'variant']}**"]
    b, s, pr, _ = nodes[()]
    _, kids = branches(pr[root_c], b, s)
    for tok, q, nb, ns in kids:
        h = ((root_c, tok, b, s),)
        if h in policy and policy[h] is not None:
            lines.append(f"  - 第 1 球若是{zh[tok]} ({q:.0%}) → 球數變成 {nb}-{ns}，第 2 球：**{cand.at[policy[h], 'variant']}**")
    return lines


END_ZH = {"K": "三振", "BB": "保送", "soft": "弱擊", "hard": "強擊", "unfinished": "5 球後未結束",
          "pruned": "搜尋時未展開"}


def enumerate_paths(nodes, policy, cand):
    """照最佳應變策略走，列出所有可能的路徑與機率。

    每條路徑 = 每一球 (球數、投什麼、打者反應)，結尾是打席結果。
    未滿兩好球的擦棒被捕併入界外；兩好球時的看好球 / 揮空 / 擦棒被捕 = 三振。
    """
    paths = []

    def rec(h, prob, steps):
        n = len(steps) + 1
        if h not in nodes or policy.get(h) is None:
            paths.append((prob, steps, "pruned" if len(steps) < BUDGET else "unfinished"))
            return
        c = policy[h]
        b, s, pr, _ = nodes[h]
        head = f"第 {n} 球 ({b}-{s}) {cand.at[c, 'variant']}"
        end, kids = branches(pr[c], b, s)
        for k, v in end.items():
            if v > 0:
                reaction = {"K": "好球 → **三振**", "BB": "壞球 → **保送**", "soft": "打進場 → **弱擊**",
                            "hard": "打進場 → **強擊**"}[k]
                paths.append((prob * v, steps + [f"{head} → {reaction}"], k))
        for tok, q, nb, ns in kids:
            step = steps + [f"{head} → {ZH_TOK[tok]} → {nb}-{ns}"]
            if len(step) == BUDGET:
                paths.append((prob * q, step[:-1] + [step[-1] + " (**5 球用完，打席未結束**)"], "unfinished"))
            else:
                rec(h + ((c, tok, b, s),), prob * q, step)

    rec((), 1.0, [])
    return paths


def serialize_policy_tree(nodes, policy, cand):
    """Serialize the chosen adaptive policy as a compact, UI-friendly tree."""
    terminal_names = {"K": "三振", "BB": "保送", "soft": "弱擊", "hard": "強擊"}

    def terminal(result, probability=None):
        item = {"type": "terminal", "result": result}
        if probability is not None:
            item["probability"] = float(probability)
        return item

    def rec(hist, depth):
        if hist not in nodes or policy.get(hist) is None:
            return terminal("搜尋未展開")
        choice = int(policy[hist])
        balls, strikes, probs, _ = nodes[hist]
        end, kids = branches(probs[choice], balls, strikes)
        children = []
        for result, prob in end.items():
            if prob > 0:
                children.append({
                    "reaction": terminal_names[result],
                    "probability": float(prob),
                    "target": terminal(terminal_names[result]),
                })
        for reaction, prob, next_balls, next_strikes in kids:
            next_hist = hist + ((choice, reaction, balls, strikes),)
            target = terminal("5 球後未結束") if depth >= BUDGET else rec(next_hist, depth + 1)
            children.append({
                "reaction": ZH_TOK[reaction],
                "probability": float(prob),
                "next_count": f"{next_balls}-{next_strikes}",
                "target": target,
            })
        return {
            "type": "decision", "depth": depth, "count": f"{balls}-{strikes}",
            "pitch": cand.at[choice, "variant"], "children": children,
        }

    return rec((), 1)


def path_report(nodes, policy, cand, top=5):
    paths = enumerate_paths(nodes, policy, cand)
    totals = {}
    for p, _, e in paths:
        totals[e] = totals.get(e, 0.0) + p
    lines = ["**照最佳應變策略投，打席結果分布** (樹狀搜尋的精確計算；未展開的分支單獨列出)：", "",
             "| 結果 | 機率 |", "|---|---|"]
    for e in ["K", "BB", "hard", "soft", "unfinished", "pruned"]:
        if e in totals:
            lines.append(f"| {END_ZH[e]} | {totals[e]:.1%} |")
    for e, title in [("K", "最可能的三振路徑"), ("hard", "最可能被強擊的路徑"), ("BB", "最可能保送的路徑"),
                     ("unfinished", "最可能 5 球後仍未結束的路徑")]:
        sel = sorted([x for x in paths if x[2] == e], key=lambda x: -x[0])[:top]
        if not sel:
            continue
        lines += ["", f"**{title}** (前 {len(sel)} 條，合計佔該結果 {sum(x[0] for x in sel) / totals[e]:.0%})：", ""]
        for i, (p, steps, _) in enumerate(sel, 1):
            lines.append(f"{i}. **{p:.1%}**：" + "；".join(steps))
        lines.append("")
        lines.append("格式：第幾球 (投球前的球數) 投什麼 → 打者反應 → 投球後的球數或打席結果")
    return lines


def run(run_dir: Path, device: str, out_dir: Path | None = None):
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "k_sequence"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    st = Setup(run_dir, device)
    khist = k_hist_table()
    prow = st.pitcher_rows()
    pt = st.pitcher_template(prow)
    log(f"{PITCHER[0]} ({PITCHER[1]}) {SEASON}: {len(prow):,} 球，速球基準 {pt['own_fb']:.1f} mph")

    summary, curves = [], {}
    for name, bid in BATTERS:
        st.cache.clear()
        st.n_evals = 0
        bt = st.batter_template(bid, pt["p_throws"])
        cand = st.candidates(prow, bt["stand"])
        hand = "右打" if bt["stand"] == "R" else "左打"
        _, vs_res, vs_curve, vs_n = actual_outcomes(bt["stand"], bid, "batter", [2024, 2025, 2026])
        k_rate, oth_res, oth_curve, oth_n = actual_outcomes(bt["stand"], bid, "others", [SEASON])
        threshold = THRESHOLD_MULT * k_rate
        log(f"== {name} (站位 {bt['stand']})：候選 {len(cand)} 種，門檻 {threshold:.1%} (實際三振率 {k_rate:.1%} x {THRESHOLD_MULT}) ==")

        nodes = adaptive_tree(st, cand, bt, pt, khist)
        tree_value, policy = solve(nodes, BUDGET)
        seq, _, _, fixed_ranked = fixed_sequence(st, cand, bt, pt, khist)
        _, greedy_policy = solve(nodes, BUDGET, choose="greedy", khist=khist)
        log("    模擬各策略")
        sims = {
            "最佳應變策略": simulate(st, cand, bt, pt, "policy", rng, khist, policy=policy),
            "最佳固定序列": simulate(st, cand, bt, pt, "fixed", rng, khist, seq=seq),
            "貪婪策略": simulate(st, cand, bt, pt, "greedy", rng, khist),
            "投手實際傾向": simulate(st, cand, bt, pt, "count", rng, khist),
            "隨機配球": simulate(st, cand, bt, pt, "usage", rng, khist),
        }
        strategies = {**sims,
                      f"實際：對 {name} (2024-2026，{vs_n} 個打席)": (vs_res, vs_curve),
                      f"實際：對其他{hand} ({SEASON}，{oth_n} 個打席)": (oth_res, oth_curve)}
        log(f"    樹狀搜尋估計的 5 球內三振 (下界) {tree_value['K']:.1%}；模擬 {sims['最佳應變策略'][1][-1]:.1%}")
        curves[name] = (strategies, threshold, bt["stand"])
        for sname, (res, curve) in strategies.items():
            reach = next((i + 1 for i, v in enumerate(curve) if not np.isnan(v) and v >= threshold), None)
            summary.append({"打者": name, "站位": bt["stand"], "策略": sname,
                            **{f"{n}球內三振": curve[n - 1] for n in range(1, BUDGET + 1)},
                            "5球內保送": res["BB"] if res else np.nan, "5球內強擊": res["hard"] if res else np.nan,
                            "5球內弱擊": res["soft"] if res else np.nan, "5球後未結束": res["unfinished"] if res else np.nan,
                            "門檻": threshold, "達到門檻的球數": reach})
        write_batter(out, name, bt, cand, strategies, threshold, nodes, policy, seq, k_rate, st.n_evals,
                     p_throws=pt["p_throws"],
                     greedy_policy=greedy_policy, fixed_ranked=fixed_ranked)

    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    plot_curves(curves, out / "k_within_curves.png")
    write_summary(out, table)
    log(f"-> {out}")


def sim_path_lines(res, top=5):
    lines = []
    for e, title in [("K", "最常出現的三振路徑"), ("hard", "最常出現的強擊路徑")]:
        sel = [x for x in res["paths"] if x[2] == e][:top]
        if sel:
            lines += [f"**{title}** (模擬 {MC_N:,} 個打席中出現的比例)：", ""]
            lines += [f"{i}. **{p:.2%}** ({round(p * MC_N)} 次)：{path}" for i, (p, path, _) in enumerate(sel, 1)]
            lines.append("")
    return lines


def write_batter(out, name, bt, cand, strategies, threshold, nodes, policy, seq, k_rate, n_evals,
                 p_throws=None, greedy_policy=None, fixed_ranked=None):
    pct = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"
    lines = [f"# {PITCHER[0]} vs {name} (站位 {bt['stand']})", "", f"情境：{CONTEXT_TEXT}", "",
             f"門檻 = {PITCHER[0]} {SEASON} 年對其他{'右' if bt['stand'] == 'R' else '左'}打 (排除 {name}) 實際三振率 {k_rate:.1%} x {THRESHOLD_MULT} = **{threshold:.1%}**",
             f"模型評估次數：{n_evals:,}；策略比較的數字皆由模擬 {MC_N:,} 個打席得到 (實際結果除外)", "", "## 策略比較", "",
             "| 策略 | 1 球內 | 2 球內 | 3 球內 | 4 球內 | **5 球內三振** | 保送 | 強擊 | 弱擊 | 5 球後未結束 | 達到門檻 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sname, (res, curve) in strategies.items():
        reach = next((f"{i + 1} 球" for i, v in enumerate(curve) if not np.isnan(v) and v >= threshold), "未達到")
        lines.append(f"| {sname} | " + " | ".join(pct(v) for v in curve[:4]) + f" | **{pct(curve[4])}** | "
                     + " | ".join(pct(res[k]) if res else "-" for k in ("BB", "hard", "soft", "unfinished")) + f" | {reach} |")
    lines += ["", "## 最佳應變策略", "",
              "策略是一棵樹：每一球依打者上一球的反應決定。「每一步都走最可能的反應」得到的單一路徑",
              "通常不會以三振結束 (例如 0-2 時最常見的反應是壞球)，所以改為列出各種結果最可能的路徑。", "",
              "**前兩球的應變**：", *describe_policy(nodes, policy, cand), "",
              *path_report(nodes, policy, cand), "",
              "## 最佳固定序列", "", "不管打者怎麼反應，都照這個順序投。前 10 名 (5 球內三振機率由樹狀計算，與上表模擬值可能略有差異)：", "",
              "| 排名 | 5 球內三振 | 第 1 球 | 第 2 球 | 第 3 球 | 第 4 球 | 第 5 球 |", "|---|---|---|---|---|---|---|",
              *[f"| {i} | {k:.1%} | " + " | ".join(cand.at[c, "variant"] for c in sq) + " |"
                for i, (sq, k) in enumerate(fixed_ranked or [], 1)], "",
              *sim_path_lines(strategies["最佳固定序列"][0]),
              "## 貪婪策略", "", "每一球只挑「這一球的三振機率 + 球數推進後的歷史三振率」最高的，不往後看。", "",
              "**前兩球的應變**：", *describe_policy(nodes, greedy_policy, cand), "",
              *sim_path_lines(strategies["貪婪策略"][0]),
              "## 投手實際傾向", "", f"依{PITCHER[0]} {SEASON} 年在各球數的實際配球比例隨機投。", "",
              *sim_path_lines(strategies["投手實際傾向"][0]),
              "## 隨機配球", "", f"依{PITCHER[0]} {SEASON} 年整體的配球比例隨機投。", "",
              *sim_path_lines(strategies["隨機配球"][0]),
              "## 候選球 (每組一顆代表性的真實球)", "",
              "| 球種_位置 | 投球數 | 使用比例 | 球速 | 轉速 | 水平位移 (打者視角) | 垂直位移 |", "|---|---|---|---|---|---|---|"]
    for r in cand.sort_values("n", ascending=False).itertuples():
        lines.append(f"| {r.variant} | {r.n} | {r.usage:.1%} | {r.release_speed:.1f} | {r.release_spin_rate:.0f} | {r.pfx_x_bv:+.2f} | {r.pfx_z:+.2f} |")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or str(bt["batter"])
    report = "\n".join(lines) + "\n"
    (out / f"{slug}.md").write_text(report)

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items() if k != "paths"}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return value

    strategy_json = []
    for sname, (res, curve) in strategies.items():
        top_paths = {}
        if res and "paths" in res:
            for ending in ("K", "hard"):
                top_paths[ending] = [
                    {"probability": float(prob), "sequence": path}
                    for prob, path, end in res["paths"] if end == ending
                ][:5]
        strategy_json.append({
            "name": sname,
            "curve": clean(curve),
            "outcomes": clean(res),
            "top_paths": top_paths,
            "is_actual": sname.startswith("實際"),
        })

    def policy_preview(selected_policy):
        if not selected_policy or () not in selected_policy:
            return None
        root = int(selected_policy[()])
        b, s, probs, _ = nodes[()]
        _, kids = branches(probs[root], b, s)
        responses = []
        for tok, prob, nb, ns in kids:
            hist = ((root, tok, b, s),)
            choice = selected_policy.get(hist)
            responses.append({
                "reaction": ZH_TOK[tok], "probability": float(prob),
                "count": f"{nb}-{ns}",
                "next_pitch": cand.at[int(choice), "variant"] if choice is not None else None,
            })
        return {"first_pitch": cand.at[root, "variant"], "responses": responses}

    actual_label = next((s["name"] for s in strategy_json if s["name"].startswith("實際：對其他")), None)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pitcher": {"id": int(PITCHER[1]), "name": PITCHER[0], "throws": p_throws},
        "batter": {"id": int(bt["batter"]), "name": name, "stand": bt["stand"]},
        "season": SEASON,
        "budget": BUDGET,
        "context": CONTEXT_TEXT,
        "threshold": clean(float(threshold)),
        "baseline_k_rate": clean(float(k_rate)),
        "model_evaluations": int(n_evals),
        "strategies": strategy_json,
        "actual_baseline_strategy": actual_label,
        "policy_preview": {
            "adaptive": policy_preview(policy),
            "greedy": policy_preview(greedy_policy),
        },
        "policy_tree": serialize_policy_tree(nodes, policy, cand),
        "fixed_sequences": [
            {"rank": i, "k_probability": float(k),
             "pitches": [cand.at[int(c), "variant"] for c in sq]}
            for i, (sq, k) in enumerate(fixed_ranked or [], 1)
        ],
        "candidates": clean(cand.sort_values("n", ascending=False).to_dict(orient="records")),
        "report_file": f"{slug}.md",
    }
    (out / f"{slug}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def plot_curves(curves, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 5.2), sharey=True)
    for ax, (name, (strategies, threshold, stand)) in zip(np.atleast_1d(axes), curves.items()):
        for sname, (_, curve) in strategies.items():
            style = "--" if sname.startswith("實際") else "-"
            ax.plot(range(1, BUDGET + 1), curve, style, marker="o", label=sname)
        ax.axhline(threshold, color="red", lw=1, ls=":", label=f"門檻 {threshold:.0%}")
        ax.set_title(f"{PITCHER[0]} vs {name} ({stand})")
        ax.set_xlabel("最多投幾球")
        ax.set_xticks(range(1, BUDGET + 1))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("幾球內三振的機率")
    np.atleast_1d(axes)[-1].legend(fontsize=9, loc="upper left")
    fig.suptitle(f"限制球數內的三振機率｜{CONTEXT_TEXT}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_summary(out, table):
    pct = lambda v: "-" if pd.isna(v) else f"{v:.1%}"
    lines = [f"# {PITCHER[0]}：5 球內三振機率｜{SEASON}", "", f"情境：{CONTEXT_TEXT}", "",
             "| 打者 | 策略 | 5 球內三振 | 保送 | 強擊 | 5 球後未結束 | 達到門檻的球數 |", "|---|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        reach = "未達到" if pd.isna(r[-1]) else f"{int(r[-1])} 球"
        lines.append(f"| {r[0]} ({r[1]}) | {r[2]} | **{pct(r[7])}** | {pct(r[8])} | {pct(r[9])} | {pct(r[11])} | {reach} (門檻 {pct(r[12])}) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="限制 5 球內三振機率最高的配球")
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batters", nargs="*", help="只跑這些打者 (名字需與 BATTERS 相同)，預設全部")
    ap.add_argument("--pitcher-id", type=int, help="覆寫投手 MLBAM ID")
    ap.add_argument("--pitcher-name", help="覆寫投手顯示名稱")
    ap.add_argument("--batter", action="append", metavar="ID:NAME",
                    help="指定打者，可重複使用；例如 --batter '592450:Aaron Judge'")
    ap.add_argument("--out", type=Path, help="輸出目錄（預設為 <run>/k_sequence）")
    args = ap.parse_args()
    global PITCHER, BATTERS
    if args.pitcher_id:
        PITCHER = (args.pitcher_name or str(args.pitcher_id), args.pitcher_id)
    elif args.pitcher_name:
        PITCHER = (args.pitcher_name, PITCHER[1])
    if args.batter:
        selected = []
        for value in args.batter:
            raw_id, sep, name = value.partition(":")
            if not sep or not raw_id.isdigit() or not name.strip():
                ap.error(f"--batter 格式必須是 ID:NAME，收到 {value!r}")
            selected.append((name.strip(), int(raw_id)))
        BATTERS = selected
    if args.batters:
        BATTERS = [b for b in BATTERS if b[0] in args.batters]
    if not BATTERS:
        ap.error("沒有可分析的打者")
    run(Path(args.run), args.device, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
