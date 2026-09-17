"""Zero-dependency local web server for the k-sequence explorer.

Run with::

    python -m ui.server

The analysis itself runs in a child process, so a long GPU search never blocks
the browser or the status endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from data_preparation import data_spec as spec


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "runs"
SEASON = spec.LAST_SEASON_YEAR


def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.result_paths: dict[str, Path] = {}
        self._bootstrap = None

    def bootstrap(self):
        if self._bootstrap is None:
            bio = pd.read_parquet(spec.RAW_DIR / "player_bio.parquet", columns=["player_id", "full_name"])
            names = dict(zip(bio["player_id"].astype(int), bio["full_name"]))
            pitch_path = spec.INTERIM_DIR / f"pitches_{SEASON}.parquet"
            pitches = pd.read_parquet(
                pitch_path,
                columns=["pitcher", "batter", "player_name", "stand", "p_throws", "is_model_target"],
            )
            valid = pitches[pitches["is_model_target"]]

            pitcher_rows = []
            for (player_id, fallback, throws), group in valid.groupby(["pitcher", "player_name", "p_throws"], dropna=False):
                if len(group) < 100:
                    continue
                player_id = int(player_id)
                pitcher_rows.append({
                    "id": player_id,
                    "name": names.get(player_id, str(fallback)),
                    "throws": str(throws),
                    "pitches": int(len(group)),
                })
            pitcher_rows.sort(key=lambda x: (-x["pitches"], x["name"]))

            batter_rows = []
            for player_id, group in valid.groupby("batter"):
                player_id = int(player_id)
                stands = sorted(set(group["stand"].dropna().astype(str)))
                batter_rows.append({
                    "id": player_id,
                    "name": names.get(player_id, str(player_id)),
                    "stands": stands,
                    "pitches": int(len(group)),
                })
            batter_rows.sort(key=lambda x: x["name"])

            runs = []
            for path in sorted(RUNS.iterdir(), reverse=True) if RUNS.exists() else []:
                if path.is_dir() and all((path / f).exists() for f in ("model.pt", "config.yaml", "normalizer.json")):
                    runs.append({"id": path.name, "label": path.name})
            self._bootstrap = {
                "season": SEASON,
                "pitchers": pitcher_rows,
                "batters": batter_rows,
                "runs": runs,
            }
        self.scan_results()
        return {**self._bootstrap, "results": self.result_summaries()}

    def scan_results(self):
        if not RUNS.exists():
            return
        for path in RUNS.glob("**/k_sequence/**/*.json"):
            self.register_result(path)
        for path in RUNS.glob("**/k_sequence_ui/**/*.json"):
            self.register_result(path)
        for path in RUNS.glob("*/k_sequence/*.md"):
            if path.name != "summary.md" and not path.with_suffix(".json").exists():
                self.register_result(path)

    def register_result(self, path: Path):
        key = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
        self.result_paths[key] = path
        return key

    def result_summaries(self):
        rows = []
        for key, path in self.result_paths.items():
            try:
                payload = self.load_result(key, include_report=False)
                rows.append({
                    "id": key,
                    "pitcher": payload["pitcher"]["name"],
                    "batter": payload["batter"]["name"],
                    "stand": payload["batter"].get("stand"),
                    "generated_at": payload.get("generated_at"),
                    "source": "structured" if path.suffix == ".json" else "legacy",
                })
            except Exception:
                continue
        return sorted(rows, key=lambda x: x.get("generated_at") or "", reverse=True)

    def load_result(self, key, include_report=True):
        path = self.result_paths[key]
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            report = path.with_name(payload.get("report_file", ""))
            if include_report and report.is_file():
                payload["report_markdown"] = report.read_text()
            return payload
        return parse_legacy_report(path, include_report=include_report)

    def start_job(self, body):
        boot = self.bootstrap()
        runs = {item["id"] for item in boot["runs"]}
        pitchers = {item["id"]: item for item in boot["pitchers"]}
        batters = {item["id"]: item for item in boot["batters"]}
        run_id = str(body.get("run", ""))
        pitcher_id = int(body.get("pitcher_id", 0))
        batter_id = int(body.get("batter_id", 0))
        if run_id not in runs or pitcher_id not in pitchers or batter_id not in batters:
            raise ValueError("模型、投手或打者選項無效")
        with self.lock:
            running = next((j for j in self.jobs.values() if j["status"] in ("queued", "running")), None)
        if running:
            raise ValueError(f"已有分析正在執行：{running['pitcher']} vs {running['batter']}")

        run_dir = (RUNS / run_id).resolve()
        out_dir = run_dir / "k_sequence_ui" / str(pitcher_id) / str(batter_id)
        existing = next(out_dir.glob("*.json"), None) if out_dir.exists() else None
        if existing and not body.get("force"):
            result_id = self.register_result(existing)
            return {"status": "cached", "result_id": result_id}

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "status": "queued", "progress": [], "result_id": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "pitcher": pitchers[pitcher_id]["name"], "batter": batters[batter_id]["name"],
        }
        with self.lock:
            self.jobs[job_id] = job
        command = [
            sys.executable, "-m", "analysis.k_sequence", "--run", str(run_dir),
            "--pitcher-id", str(pitcher_id), "--pitcher-name", pitchers[pitcher_id]["name"],
            "--batter", f"{batter_id}:{batters[batter_id]['name']}", "--out", str(out_dir),
        ]
        threading.Thread(target=self._run_job, args=(job_id, command, out_dir), daemon=True).start()
        return job

    def _run_job(self, job_id, command, out_dir):
        job = self.jobs[job_id]
        job["status"] = "running"
        try:
            proc = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                with self.lock:
                    job["progress"].append(line.rstrip())
                    job["progress"] = job["progress"][-400:]
            code = proc.wait()
            if code:
                detail = "\n".join(job["progress"][-8:])
                raise RuntimeError(f"分析程序結束，代碼 {code}\n{detail}")
            result = next(out_dir.glob("*.json"), None)
            if not result:
                raise RuntimeError("分析完成，但找不到 JSON 結果")
            job["result_id"] = self.register_result(result)
            job["status"] = "complete"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
        finally:
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")


def parse_legacy_report(path: Path, include_report=True):
    """Adapt pre-UI Markdown/CSV output to the current result schema."""
    text = path.read_text()
    title = re.search(r"^# (.+?) vs (.+?) \(站位 ([LR])\)", text, re.M)
    if not title:
        raise ValueError("不是 k_sequence 打者報告")
    pitcher_name, batter_name, stand = title.groups()
    summary_path = path.with_name("summary.csv")
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8-sig")))
    rows = [r for r in rows if r["打者"] == batter_name]
    strategies = []
    for row in rows:
        strategy_name = row["策略"]
        section = ""
        if not strategy_name.startswith("實際"):
            section_match = re.search(
                rf"^## {re.escape(strategy_name)}\s*$([\s\S]*?)(?=^## |\Z)", text, re.M
            )
            section = section_match.group(1) if section_match else ""
        paths = [
            {"probability": float(match.group(1)) / 100, "sequence": match.group(2)}
            for match in re.finditer(
                r"^\d+\. \*\*([\d.]+)%\*\*(?: \(\d+ 次\))?：(.*)$", section, re.M
            )
        ]
        strategies.append({
            "name": strategy_name,
            "curve": [float(row[f"{n}球內三振"]) for n in range(1, 6)],
            "outcomes": {
                "BB": float(row["5球內保送"]), "hard": float(row["5球內強擊"]),
                "soft": float(row["5球內弱擊"]), "unfinished": float(row["5球後未結束"]),
            },
            "top_paths": {"K": paths[:5]}, "is_actual": strategy_name.startswith("實際"),
        })
    fixed = []
    fixed_section = text.split("## 最佳固定序列", 1)[-1].split("## 貪婪策略", 1)[0]
    for match in re.finditer(r"^\| (\d+) \| ([\d.]+)% \| (.+?) \|$", fixed_section, re.M):
        cells = [c.strip() for c in match.group(3).split(" | ")]
        fixed.append({"rank": int(match.group(1)), "k_probability": float(match.group(2)) / 100, "pitches": cells})

    threshold = float(rows[0]["門檻"]) if rows else None
    context = re.search(r"^情境：(.+)$", text, re.M)
    payload = {
        "schema_version": 0, "generated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "pitcher": {"id": None, "name": pitcher_name, "throws": None},
        "batter": {"id": None, "name": batter_name, "stand": stand},
        "season": SEASON, "budget": 5, "context": context.group(1) if context else "",
        "threshold": threshold, "baseline_k_rate": threshold / 1.5 if threshold is not None else None,
        "strategies": strategies, "actual_baseline_strategy": next(
            (r["name"] for r in strategies if r["name"].startswith("實際：對其他")), None),
        "policy_preview": {}, "policy_tree": None, "fixed_sequences": fixed, "candidates": [],
        "legacy": True,
    }
    if include_report:
        payload["report_markdown"] = text
    return payload


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    server_version = "BaseballSequenceUI/1.0"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, data, status=HTTPStatus.OK):
        raw = json.dumps(data, ensure_ascii=False, default=_json_safe).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/bootstrap":
                return self.send_json(STATE.bootstrap())
            if path.startswith("/api/results/"):
                key = path.rsplit("/", 1)[-1]
                return self.send_json(STATE.load_result(key))
            if path.startswith("/api/jobs/"):
                key = path.rsplit("/", 1)[-1]
                return self.send_json(STATE.jobs[key])
            return self.serve_static(path)
        except KeyError:
            self.send_json({"error": "找不到指定資源"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/analyze":
            return self.send_json({"error": "找不到指定資源"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            self.send_json(STATE.start_job(body), HTTPStatus.ACCEPTED)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_static(self, request_path):
        rel = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        target = (STATIC / rel).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            target = STATIC / "index.html"
        mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}.get(
            target.suffix, "application/octet-stream"
        )
        raw = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main():
    parser = argparse.ArgumentParser(description="k_sequence 本機網頁介面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"k_sequence UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
