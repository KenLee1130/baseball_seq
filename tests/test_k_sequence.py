import json

import numpy as np
import pandas as pd

from analysis import k_sequence as ks


def test_strategy_report_includes_serialized_decision_tree(tmp_path):
    probabilities = np.array([[0.20, 0.20, 0.20, 0.05, 0.10, 0.15, 0.10]])
    nodes = {(): (0, 0, probabilities, [0])}
    policy = {(): 0}
    candidates = pd.DataFrame([{
        "variant": "四縫線_中間高", "pitch_family": "fastball", "region": "中間高",
        "n": 80, "usage": 1.0,
        "release_speed": 96.2, "release_spin_rate": 2350,
        "pfx_x_bv": -0.4, "pfx_z": 1.4, "plate_x_bv": 0.1, "plate_z_norm": 0.82,
    }])
    path = "第 1 球 (0-0) 四縫線_中間高 → 看好球 → 0-1；第 2 球 (0-1) 四縫線_中間高 → 好球 → **三振**"
    result = {"K": 0.2, "BB": 0.0, "soft": 0.15, "hard": 0.1, "unfinished": 0.55,
              "paths": [(0.2, path, "K")]}
    strategies = {
        name: (dict(result), [0.0, 0.0, 0.1, 0.15, 0.2])
        for name in ("最佳應變策略", "最佳固定序列", "貪婪策略", "投手實際傾向", "隨機配球")
    }

    ks.write_batter(
        tmp_path, "Test Batter", {"batter": 2, "stand": "R"}, candidates,
        strategies, 0.3, nodes, policy, (0, 0, 0, 0, 0), 0.2, 1,
        model_run="event7_height_test", p_throws="R", greedy_policy=policy,
        fixed_ranked=[((0, 0, 0, 0, 0), 0.2)],
    )

    payload = json.loads((tmp_path / "Test_Batter.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["model_run"] == "event7_height_test"
    assert payload["policy_tree"]["pitch"] == "四縫線_中間高"
    assert payload["policy_tree"]["count"] == "0-0"
    assert {child["reaction"] for child in payload["policy_tree"]["children"]} >= {"壞球", "揮空", "強擊"}
    assert payload["fixed_sequences"][0]["pitches"] == ["四縫線_中間高"] * 5
    assert payload["fixed_sequences"][0]["pitch_details"][0]["plate_z_norm"] == 0.82
    assert payload["strategies"][0]["top_paths"]["K"][0]["pitches"][1]["count"] == "0-1"
