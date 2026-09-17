"""
train.py
========
依 configs/*.yaml 訓練 7 類事件模型，輸出到 runs/<name>_<時間>/。

  runs/<run>/
    config.yaml        本次使用的設定 (含特徵設定與資料切分)
    model.pt           val loss 最低的權重
    history.jsonl      每個 epoch 的 train / val loss
    train.log          訓練過程
    eval_val/          訓練結束後自動在 val 上評估 (evaluate.py)

資料：splits.SPLITS 的 train / val 球季，讀 features.py 產生的快取張量，整份放上 GPU。

用法:
    python -m modeling.train --config configs/event7_base.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_preparation"))

import splits as sp                                      # noqa: E402
from modeling import features as F                       # noqa: E402
from modeling.model import ModelConfig, PitchSequenceModel  # noqa: E402

RUNS_DIR = ROOT / "runs"
INPUTS = ["token_num", "token_cat", "pad", "ctx_num", "ctx_cat"]
COUNT_SLOT = F.CTX_CAT.index("count_state")


class Logger:
    def __init__(self, path: Path):
        self.f = path.open("a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def load_split_tensors(split: str, device: torch.device) -> dict[str, torch.Tensor]:
    """把某一切分所有球季的快取張量串接後放上裝置。只保留有 7 類標籤的樣本。"""
    parts: dict[str, list[np.ndarray]] = {}
    for season in sp.SPLITS[split]:
        arr = F.load_cached(season, mmap=True)
        keep = np.asarray(arr["m_event"])
        for k in INPUTS + ["y_event"]:
            parts.setdefault(k, []).append(np.asarray(arr[k])[keep])
    out = {}  # token_num 保持 float16 以節省 GPU 記憶體，每個 batch 再轉 float32
    for k, v in parts.items():
        x = torch.from_numpy(np.concatenate(v))
        if k in ("token_cat", "ctx_cat", "y_event"):
            x = x.long()
        out[k] = x.to(device)
    return out


def batches(n: int, batch_size: int, shuffle: bool, device: torch.device):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


def run_epoch(model, data, cfg_train, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, count = 0.0, 0
    n = len(data["y_event"])
    with torch.set_grad_enabled(training):
        for b in batches(n, cfg_train["batch_size"], shuffle=training, device=device):
            x = [data[k][b] for k in INPUTS]
            x[0] = x[0].float()                          # token_num 以 float16 存在 GPU 上
            if training and cfg_train["count_dropout"] > 0:
                ctx_cat = x[4].clone()
                drop = torch.rand(len(b), device=device) < cfg_train["count_dropout"]
                ctx_cat[drop, COUNT_SLOT] = 0          # 0 = <pad>，等於「不告訴模型球數」
                x[4] = ctx_cat
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(*x)
            loss = loss_fn(logits.float(), data["y_event"][b])
            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss / len(b)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            count += len(b)
    return total / count


def main() -> int:
    ap = argparse.ArgumentParser(description="訓練 7 類事件模型")
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-eval", action="store_true", help="訓練完不自動評估")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")

    run_dir = RUNS_DIR / f"{cfg['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    log = Logger(run_dir / "train.log")

    info = json.loads((F.ARRAY_DIR / str(sp.SPLITS["train"][0]) / "info.json").read_text())
    cfg["features"] = info["feature_config"]
    cfg["feature_names"] = info["feature_names"]
    cfg["splits"] = sp.SPLITS
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    shutil.copy(F.NORMALIZER_PATH, run_dir / "normalizer.json")

    log(f"run: {run_dir.name} | device: {device}")
    t0 = time.time()
    train = load_split_tensors("train", device)
    val = load_split_tensors("val", device)
    log(f"資料: train {len(train['y_event']):,} 樣本 ({sp.SPLITS['train']}) | "
        f"val {len(val['y_event']):,} 樣本 ({sp.SPLITS['val']}) | 載入 {time.time() - t0:.0f}s")

    model = PitchSequenceModel(train["token_num"].shape[-1], train["ctx_num"].shape[-1],
                               ModelConfig(**cfg["model"])).to(device)
    log(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    # 基準：只用 train 的類別比例猜 (不看任何特徵) 的 log loss
    prior = torch.bincount(train["y_event"], minlength=7).float()
    prior = prior / prior.sum()
    baseline = float(-(torch.log(prior)[val["y_event"]]).mean())
    log(f"基準 val loss (只用類別比例): {baseline:.4f}")

    best, wait = float("inf"), 0
    history = (run_dir / "history.jsonl").open("w")
    for epoch in range(1, tc["max_epochs"] + 1):
        t0 = time.time()
        tl = run_epoch(model, train, tc, device, optimizer)
        vl = run_epoch(model, val, tc, device)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]["lr"]
        improved = vl < best - tc["min_delta"]
        log(f"epoch {epoch:02d} | train {tl:.4f} | val {vl:.4f} (比基準低 {(1 - vl / baseline):.2%}) "
            f"| lr {lr:.1e} | {time.time() - t0:.0f}s{' *' if improved else ''}")
        history.write(json.dumps({"epoch": epoch, "train_loss": tl, "val_loss": vl, "lr": lr}) + "\n")
        history.flush()
        if improved:
            best, wait = vl, 0
            torch.save({"state_dict": model.state_dict(), "model_config": cfg["model"],
                        "n_token_num": train["token_num"].shape[-1], "n_ctx_num": train["ctx_num"].shape[-1],
                        "epoch": epoch, "val_loss": vl, "baseline_val_loss": baseline}, run_dir / "model.pt")
        else:
            wait += 1
            if wait >= tc["patience"]:
                log("early stopping")
                break
    log(f"最佳 val loss {best:.4f} -> {run_dir / 'model.pt'}")

    del train, val
    torch.cuda.empty_cache()
    if not args.no_eval:
        from modeling import evaluate
        evaluate.evaluate_run(run_dir, "val", log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
