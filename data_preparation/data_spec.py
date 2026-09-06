"""
data_spec.py
============
單一事實來源 (single source of truth)：本專案要抓哪些資料、每個欄位的用途、
以及三段式預測目標的精確定義。

分工：
  - data_spec.py   只描述「要什麼」，不做任何 I/O、不做任何計算。
  - fetch_data.py  照這份規格把原始資料抓下來，存進 dataset/raw/。
  - preprocess.py  (之後) 依 DERIVED_FEATURES 計算衍生特徵、正規化、切 train/test。

研究問題：
  打者可以依「對球的反應」分群；對每一群，找出最有效的配球「序列」
  (不只是單一球種/位置，而是前幾球如何鋪陳)。

所有事實均以 2025 年實際 Statcast 回傳資料驗證過 (pybaseball 2.2.7)。
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
DATASET_DIR = PACKAGE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"          # fetch_data.py 的輸出：未經加工的原始資料
INTERIM_DIR = DATASET_DIR / "interim"  # preprocess 的中間產物 (衍生特徵)
SPLIT_DIR = DATASET_DIR / "splits"     # train / test


# ---------------------------------------------------------------------------
# 賽季設定
# ---------------------------------------------------------------------------
# 刻意用兩個不同賽季，避免資料洩漏 (data leakage)：
#   打者分群的「反應輪廓」用 PROFILE_SEASON 建立，
#   配球序列模型則在 MODEL_SEASON 上訓練與評估。
#   如此一來，「這個打者屬於哪一群」不會偷看到他在建模賽季的表現。

SEASONS: dict[str, dict[str, str]] = {
    # 打者輪廓 / 分群用：上一個完整賽季
    "profile": {"year": "2024", "start": "2024-03-20", "end": "2024-10-01"},
    # 配球序列建模用
    "model": {"year": "2025", "start": "2025-03-18", "end": "2025-10-01"},
}

# 只保留例行賽。S=春訓, E=表演賽, A=明星賽, D/F/L/W=季後賽各輪。
# 季後賽的配球策略與例行賽差異大 (投手配置、緊張度)，先排除，需要時再單獨分析。
KEEP_GAME_TYPES = ("R",)


# ---------------------------------------------------------------------------
# Statcast 原始欄位
# ---------------------------------------------------------------------------
# 來源：pybaseball.statcast() -> Baseball Savant，一列一球，共 119 欄。
# 這裡只列出本專案要保留的欄位，依「角色」分組。
# 每一組的 key 是欄位名，value 是它在本專案裡的用途。

STATCAST_COLUMNS: dict[str, dict[str, str]] = {
    # -- 主鍵與序列骨架 -----------------------------------------------------
    # game_pk + at_bat_number + pitch_number 三者唯一決定一球，
    # 也是重建「這個打席前面投了哪些球」的唯一依據。
    "keys": {
        "game_pk": "比賽唯一 ID",
        "game_date": "比賽日期",
        "game_year": "賽季",
        "game_type": "比賽性質 (R=例行賽)，用於過濾",
        "at_bat_number": "該場比賽第幾個打席 (跨兩隊連號)",
        "pitch_number": "該打席第幾球",
    },
    # -- 參與者 -------------------------------------------------------------
    "participants": {
        "batter": "打者 MLBAM ID",
        "pitcher": "投手 MLBAM ID (序列效應的固定效應必須用它)",
        "player_name": "投手姓名 (注意：不是打者)",
        "fielder_2": "捕手 MLBAM ID，配球實際上是捕手在配",
        "stand": "打者站位 L/R (左右開弓者顯示本打席實際站位)",
        "p_throws": "投手慣用手 L/R",
        "home_team": "主隊",
        "away_team": "客隊",
        "inning": "局數",
        "inning_topbot": "上下半局，用於判斷進攻方",
    },
    # -- 球的內容 -----------------------------------------------------------
    # 轉速本身不是好的球質代理，位移 (pfx) 與延伸 (extension) 才是。
    "pitch_content": {
        "pitch_type": "球種代碼 (FF/SI/SL/ST/CH/FC/CU/FS/KC/SV...)",
        "pitch_name": "球種全名",
        "release_speed": "出手球速 (mph)",
        "effective_speed": "考慮延伸後的等效球速 (mph)",
        "release_spin_rate": "轉速 (rpm)，須在球種內比較",
        "spin_axis": "轉軸角度 (度)",
        "pfx_x": "水平位移 (ft)",
        "pfx_z": "垂直位移 (ft)",
        "api_break_x_arm": "投手臂側方向水平變化量",
        "api_break_z_with_gravity": "含重力的垂直落差",
        "release_pos_x": "出手點水平位置 (ft)，tunneling 特徵",
        "release_pos_z": "出手點高度 (ft)，tunneling 特徵",
        "release_pos_y": "出手點與本壘板距離 (ft)",
        "release_extension": "延伸距離 (ft)",
        "arm_angle": "手臂角度 (度)",
        "vx0": "初速度 x 分量",
        "vy0": "初速度 y 分量",
        "vz0": "初速度 z 分量",
        "ax": "加速度 x 分量",
        "ay": "加速度 y 分量",
        "az": "加速度 z 分量",
    },
    # -- 進壘位置 -----------------------------------------------------------
    # plate_z 必須用該打者自己的好球帶正規化 (見 DERIVED_FEATURES)，
    # 做完之後身高幾乎不再帶有額外資訊。
    "location": {
        "plate_x": "通過本壘板時的水平位置 (ft，捕手視角)",
        "plate_z": "通過本壘板時的高度 (ft)",
        "zone": "Savant 區塊編號 1-14 (1-9 好球帶內, 11-14 外)",
        "sz_top": "該打者好球帶上緣 (ft)",
        "sz_bot": "該打者好球帶下緣 (ft)",
    },
    # -- 情境 ---------------------------------------------------------------
    "context": {
        "balls": "壞球數 0-3",
        "strikes": "好球數 0-2",
        "outs_when_up": "出局數 0-2",
        "on_1b": "一壘跑者 ID (無人為 NaN)",
        "on_2b": "二壘跑者 ID",
        "on_3b": "三壘跑者 ID",
        "bat_score": "進攻方分數",
        "fld_score": "守備方分數",
        "bat_score_diff": "進攻方分差",
        "n_thruorder_pitcher": "投手第幾輪面對打線 (times through order)",
        "n_priorpa_thisgame_player_at_bat": "打者本場先前打席數 (不分投手)",
        "if_fielding_alignment": "內野佈陣",
        "of_fielding_alignment": "外野佈陣",
        "age_bat": "打者年齡",
        "age_pit": "投手年齡",
        "pitcher_days_since_prev_game": "投手距上次出賽天數 (疲勞代理)",
        "batter_days_since_prev_game": "打者距上次出賽天數",
    },
    # -- 打者反應：三段式目標的來源 ----------------------------------------
    # bat_speed / swing_length 連「揮空」的球都有值 (實測 99%)，
    # 所以它們能描述揮棒本身，而不只是描述觸球結果。
    "batter_reaction": {
        "description": "每一球的結果字串，三段式目標全部由它定義",
        "type": "簡化結果 B=ball, S=strike, X=in play",
        "events": "打席終結事件 (只有打席最後一球有值)",
        "des": "文字敘述",
        "bat_speed": "棒速 (mph)，揮棒即有值",
        "swing_length": "揮棒軌跡長度 (ft)，揮棒即有值",
        "attack_angle": "揮棒攻擊角 (度) [2025 新增]",
        "attack_direction": "揮棒方向 (度) [2025 新增]",
        "swing_path_tilt": "揮棒平面傾角 (度) [2025 新增]",
        "intercept_ball_minus_batter_pos_x_inches": "擊球點相對打者位置 x (吋)，等於揮棒時機 [2025 新增]",
        "intercept_ball_minus_batter_pos_y_inches": "擊球點相對打者位置 y (吋) [2025 新增]",
    },
    # -- 擊球結果 -----------------------------------------------------------
    "batted_ball": {
        "launch_speed": "擊球初速 (mph) <- 第三段目標的主變數",
        "launch_angle": "仰角 (度)",
        "launch_speed_angle": "擊球分類 1-6 (6=barrel)",
        "hit_distance_sc": "飛行距離 (ft)",
        "bb_type": "擊球型態 ground_ball/line_drive/fly_ball/popup",
        "hc_x": "落點 x，用於計算噴射角與拉打率",
        "hc_y": "落點 y",
        "hit_location": "接球守備位置",
        "estimated_ba_using_speedangle": "xBA",
        "estimated_woba_using_speedangle": "xwOBA，第三段的替代目標",
        "estimated_slg_using_speedangle": "xSLG",
        "woba_value": "實際 wOBA 值",
        "woba_denom": "wOBA 分母",
        "babip_value": "BABIP 指示值",
        "iso_value": "ISO 指示值",
    },
    # -- 價值指標 -----------------------------------------------------------
    # 每一球都有值 (含壞球)，適合當「這個配球序列值多少分」的整體評估。
    "value": {
        "delta_run_exp": "這一球造成的預期得分變化 (進攻方視角)",
        "delta_pitcher_run_exp": "投手視角的預期得分變化",
        "delta_home_win_exp": "主隊勝率變化",
    },
}

# 攤平成 fetch_data.py 直接可用的欄位清單
ALL_STATCAST_COLUMNS: list[str] = [
    col for group in STATCAST_COLUMNS.values() for col in group
]


# ---------------------------------------------------------------------------
# 球種分族
# ---------------------------------------------------------------------------
# 原始球種有 16 種以上，直接做序列會讓組合數爆炸、樣本過薄。
# 收斂成 5 族：序列詞彙必須夠粗，統計上才估得動。

PITCH_FAMILY: dict[str, str] = {
    "FF": "fastball",     # four-seam
    "FA": "fastball",     # 泛稱直球
    "FC": "cutter",       # cutter 動作介於速球與滑球，單獨一族
    "SI": "sinker",
    "FT": "sinker",       # 舊代碼 two-seam
    "SL": "slider",
    "ST": "slider",       # sweeper
    "SV": "slider",       # slurve
    "CU": "curveball",
    "KC": "curveball",    # knuckle-curve
    "CS": "curveball",    # slow curve
    "CH": "changeup",
    "FS": "changeup",     # splitter
    "FO": "changeup",     # forkball
    "SC": "changeup",     # screwball
}

# 非競技投球，一律剔除
EXCLUDE_PITCH_TYPES: tuple[str, ...] = (
    "PO",  # 牽制
    "IN",  # 故意四壞
    "EP",  # eephus，樣本極少且性質特殊
    "KN",  # 蝴蝶球，投手極少且機制不同
    "UN",  # unknown
)


# ---------------------------------------------------------------------------
# 三段式目標 (hurdle model)
# ---------------------------------------------------------------------------
# 為什麼分三段：擊球初速是「條件式」存在的變數，沒揮棒、沒碰到球就沒有值。
# 若把沒觸球的球一律填 0，會把「投出好球讓打者看著不敢揮」(投手成功)
# 和「打者打出軟弱滾地球」(也是投手成功) 混成同一件事，模型學不到東西。
#
#   Stage 1  P(swing)              全部競技投球
#   Stage 2  P(contact | swing)    只在有揮棒的球上訓練
#   Stage 3  E[exit_velo | contact] 只在有觸球且測得初速的球上訓練
#
# 使用者定義：品質一律以擊球初速衡量。理由是 105 mph 的滾地球仍是好的擊球，
# 界外球至少代表打者跟得上，只是時間稍早或稍晚。故界外球「算觸球」。

# 有揮棒 (含觸擊)
SWING_DESCRIPTIONS: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "bunt_foul_tip",
    "missed_bunt",
    "hit_into_play",
})

# 揮了但沒碰到
WHIFF_DESCRIPTIONS: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
})

# 揮了且碰到 (Stage 2 的正例)
CONTACT_DESCRIPTIONS: frozenset[str] = SWING_DESCRIPTIONS - WHIFF_DESCRIPTIONS

# 沒揮棒
TAKE_DESCRIPTIONS: frozenset[str] = frozenset({
    "ball",
    "blocked_ball",
    "called_strike",
    "hit_by_pitch",
})

# 非競技投球 / 非打者決策，建模前剔除
NON_COMPETITIVE_DESCRIPTIONS: frozenset[str] = frozenset({
    "pitchout",
    "automatic_ball",    # 投球計時違規、故意四壞
    "automatic_strike",  # 打擊計時違規
})

# 觸擊球，打者決策機制與一般揮棒完全不同，建模前剔除
BUNT_DESCRIPTIONS: frozenset[str] = frozenset({
    "foul_bunt",
    "bunt_foul_tip",
    "missed_bunt",
})

# 實測：有觸球的球裡，只有這兩類真的量得到 launch_speed
#   hit_into_play  99.7% 有值
#   foul           87.0% 有值
#   foul_tip        0.0% 有值  <-- 觸球但無初速，Stage 3 只能當缺值
EV_MEASURED_DESCRIPTIONS: frozenset[str] = frozenset({"hit_into_play", "foul"})

TARGETS: dict[str, dict[str, object]] = {
    "stage1_swing": {
        "type": "binary",
        "positive": "description in SWING_DESCRIPTIONS",
        "population": "所有競技投球",
        "note": "打者有沒有出棒。追打壞球本身就是投手的勝利。",
    },
    "stage2_contact": {
        "type": "binary",
        "positive": "description in CONTACT_DESCRIPTIONS",
        "population": "description in SWING_DESCRIPTIONS",
        "note": "揮了有沒有碰到。界外球算碰到。",
    },
    "stage3_exit_velocity": {
        "type": "regression",
        "value": "launch_speed",
        "population": "description in EV_MEASURED_DESCRIPTIONS 且 launch_speed 非缺值",
        "note": (
            "碰到的球有多扎實。foul_tip 是觸球但量不到初速，"
            "在此段視為缺值排除，不可填 0。"
        ),
    },
}

# 輔助目標：不當主目標，但用來檢查三段式結論是否與整體價值一致
AUXILIARY_TARGETS: dict[str, str] = {
    "delta_run_exp": "每球的預期得分變化，可驗證序列建議是否真的降低失分",
    "estimated_woba_using_speedangle": "同時吃初速與仰角，作為 Stage 3 的穩健性對照",
}


# ---------------------------------------------------------------------------
# 衍生特徵 (由 preprocess 計算，不在 fetch 階段做)
# ---------------------------------------------------------------------------
# 這裡先把「要算什麼、怎麼算、為什麼」寫死，避免之後實作時漂移。

DERIVED_FEATURES: dict[str, dict[str, str]] = {
    # -- 打者靜態資訊 -------------------------------------------------------
    "batter_static": {
        "lineup_slot": (
            "該場棒次。推導：同一場同一進攻方的打席依 at_bat_number 排序，"
            "序號 mod 9 + 1。自動處理代打。"
        ),
        "height_in": "身高 (吋)，來自 MLB Stats API，非 Statcast 欄位",
        "weight_lb": "體重 (磅)，來自 MLB Stats API",
        "bats": "慣用打擊邊 L/R/S，來自 MLB Stats API",
    },
    # -- 對戰脈絡 -----------------------------------------------------------
    "matchup_context": {
        "nth_pa_vs_pitcher": (
            "本場面對這位投手的第幾個打席。推導：(game_pk, batter, pitcher) "
            "累計打席序號。注意內建的 n_priorpa_thisgame_player_at_bat 不分投手。"
        ),
        "pitcher_pitch_count": (
            "投手本場累計投球數。推導：(game_pk, pitcher) 依序累計。疲勞代理。"
        ),
        "count_state": "球數狀態，balls-strikes 當 12 個類別，不可當兩個數字",
    },
    # -- 序列特徵：本專案的核心假設 ----------------------------------------
    # 直接把前幾球的原始值丟給模型，不如把「假設本身」算成特徵：
    # 序列之所以有效，是因為速差、位置跳動、以及出手點相同但軌跡不同 (tunneling)。
    "sequence": {
        "prev{k}_pitch_family": "前 k 球的球種族 (k=1,2,3)",
        "prev{k}_plate_x": "前 k 球進壘水平位置",
        "prev{k}_plate_z_norm": "前 k 球正規化進壘高度",
        "prev{k}_release_speed": "前 k 球球速",
        "prev{k}_release_spin_rate": "前 k 球轉速",
        "prev{k}_description": "前 k 球打者反應 (揮空/看著/界外/擊入場內)",
        "prev{k}_launch_speed": "前 k 球若有觸球的初速，衡量打者跟得上與否",
        "speed_diff_prev1": "本球球速 - 前一球球速。序列效應的主要載體。",
        "plate_dist_prev1": "本球與前一球進壘點的歐氏距離 (正規化後座標)",
        "same_family_prev1": "是否與前一球同球種族",
        "release_dist_prev1": (
            "本球與前一球出手點距離。距離小但進壘點遠 = tunneling 成功。"
        ),
        "tunnel_ratio_prev1": "plate_dist_prev1 / release_dist_prev1，越大越難辨識",
        "pa_pitch_history": "本打席至今已投球種族序列 (字串)，用於序列統計",
    },
    # -- 棒球意義上的正規化 -------------------------------------------------
    # 比一般 z-score 更重要。這幾項會直接改變結論。
    "baseball_normalization": {
        "plate_z_norm": (
            "(plate_z - sz_bot) / (sz_top - sz_bot)。用打者自己的好球帶正規化高度，"
            "0=下緣 1=上緣。做完這步身高幾乎不再帶額外資訊。"
        ),
        "plate_x_norm": "plate_x / 0.7083 (好球帶半寬 ft)，-1=內角邊 1=外角邊",
        "speed_vs_own_fastball": (
            "release_speed - 該投手本季速球均速。90 mph 對某些投手是速球、"
            "對某些是變速球，只有速差有意義。"
        ),
        "spin_z_within_family": "轉速在該球種族內的 z-score，跨球種比轉速沒有意義",
        "location_from_batter_view": (
            "plate_x 依 stand 翻正，統一成「正=外角、負=內角」，"
            "否則左右打者的內外角會互相抵消。"
        ),
    },
    # -- 打者分群輪廓 (PROFILE_SEASON 上計算) -------------------------------
    # 分群特徵必須是「反應」不是「結果」。用打擊率會只分出強打者與弱打者。
    "batter_profile": {
        "swing_rate_by_family_zone": "各球種族 x 好球帶區塊的揮棒率",
        "whiff_rate_by_family_zone": "各球種族 x 區塊的揮空率",
        "chase_rate_by_family": "各球種族的追打率 (好球帶外出棒)",
        "ev_by_family": "各球種族的平均擊球初速",
        "mean_bat_speed": "平均棒速",
        "mean_swing_length": "平均揮棒長度",
        "mean_attack_angle": "平均攻擊角 [2025 起]",
        "pull_rate_by_family": (
            "各球種族的拉打率。噴射角 = degrees(arctan2(hc_x-125.42, 198.27-hc_y))，"
            "右打者 < -15 度為拉打，左打者 > 15 度為拉打。"
            "不用 FanGraphs Pull%，因為該來源目前回 403。"
        ),
        "SHRINKAGE_NOTE": (
            "所有比率特徵必須做 shrinkage 往聯盟平均收縮，權重依樣本數。"
            "某打者在某區塊只遇過 5 球就決定他的分群，比不做標準化還危險。"
        ),
    },
}

# 正規化策略：不是一律標準化，取決於模型
NORMALIZATION_POLICY: dict[str, str] = {
    "tree_models": "XGBoost/LightGBM 對單調變換不變，數值特徵餵原始值即可",
    "distance_models": "k-means/GMM/PCA/SVM 需要 z-score，因為它們算距離",
    "categorical": "球種族、zone、count_state、stand 走 one-hot，不是標準化",
    "leakage": "標準化參數只能從 train 估，再套到 test",
    "priority": "棒球意義的正規化 (見 baseball_normalization) 比 z-score 重要得多",
}


# ---------------------------------------------------------------------------
# 外部資料源 (非 Statcast)
# ---------------------------------------------------------------------------

EXTERNAL_SOURCES: dict[str, dict[str, str]] = {
    "mlb_stats_api_people": {
        "url": "https://statsapi.mlb.com/api/v1/people",
        "params": "personIds=<逗號分隔的 MLBAM ID>",
        "provides": "身高 height、體重 weight、慣用打擊邊 batSide、慣用投球手 pitchHand",
        "why": "Statcast 逐球資料沒有身高體重",
        "note": "支援批次查詢，一次數百人。無需 API key。",
    },
    "fangraphs_pull_rate": {
        "status": "UNAVAILABLE",
        "why": "pybaseball 的 FanGraphs 爬蟲目前回傳 HTTP 403 (leaders-legacy.aspx 已失效)",
        "workaround": "改由 Statcast hc_x/hc_y 自行計算噴射角與拉打率，見 batter_profile",
    },
}


# ---------------------------------------------------------------------------
# 適用範圍：亞洲職棒可移植性
# ---------------------------------------------------------------------------
# 比賽規則要求說明「結論適用範圍」。這裡先記錄哪些特徵無法移植。
# 關鍵事實：本專案的第三段目標 (擊球初速) 正好是 MLB 以外拿不到的東西。

ASIAN_LEAGUE_AVAILABILITY: dict[str, str] = {
    "sequence_pitch_family": "CPBL/NPB/KBO 皆有，可移植",
    "sequence_velocity": "皆有，可移植",
    "sequence_location": "僅粗略分區 (非精確座標)，需降級為 3x3 網格",
    "spin_rate": "無公開資料，無法移植",
    "release_point_tunneling": "無公開資料，無法移植",
    "exit_velocity": "無公開資料。Stage 3 目標必須改為結果型 (打席 wOBA)",
    "bat_speed_swing_path": "無公開資料，無法移植",
    "lineup_count_handedness": "皆有，可移植",
    "CONCLUSION": (
        "配球序列的架構可移植，但第三段目標需替換。"
        "簡報應說明：以 MLB 追蹤資料建立方法，移植中職時哪些特徵降級、目標換成什麼。"
    ),
}
