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

# Statcast 追蹤資料自 2015 年全面上線，2015 之前沒有逐球追蹤，故以此為下界。
FIRST_STATCAST_YEAR = 2015
LAST_SEASON_YEAR = 2026

# 每季的抓取視窗刻意開寬 (3/01 ~ 11/10)，不精確對齊各年開幕日。
# 理由有三：
#   1. 開幕日逐年不同 (2025 因東京開幕戰提早到 3/18，2020 因疫情縮短為 7/23~9/27)，
#      寫死日期等於每年都要維護。
#   2. 視窗外的春訓 (S) 與季後賽 (D/F/L/W) 會被 KEEP_GAME_TYPES 濾掉，開寬不會混入雜訊。
#   3. 完全沒有比賽的月份，fetch 端會寫下 .empty 標記，之後不再重複查詢。
SEASON_WINDOW = ("03-01", "11-10")


def season_range(year: int) -> dict[str, str]:
    """回傳某一年的抓取視窗。"""
    lo, hi = SEASON_WINDOW
    return {"year": str(year), "start": f"{year}-{lo}", "end": f"{year}-{hi}"}


# key 為年份字串，方便 CLI 直接用 --season 2019 指定。
SEASONS: dict[str, dict[str, str]] = {
    str(y): season_range(y)
    for y in range(FIRST_STATCAST_YEAR, LAST_SEASON_YEAR + 1)
}

# 語意別名，保留舊有用法。指向「最後一季」與「其前一季」，
# 因為打者輪廓必須由前一季建立，才不會洩漏建模賽季的資訊。
SEASON_ALIASES: dict[str, str] = {
    "profile": str(LAST_SEASON_YEAR - 1),
    "model": str(LAST_SEASON_YEAR),
}

# 欄位的可用起始年 —— 全部以實際下載的資料抽樣量測，不是憑文件推測。
# 這張表決定「哪些特徵能回溯到哪一年」，進而決定模型真正能用幾季。
#
# 量測方式：每年取一天 (7/15，若逢全明星賽或休賽期改 8/15) 看非空值比例。
# 值為 0% 者代表該欄位在當年完全不存在，不是抽樣誤差。
COLUMN_AVAILABILITY: dict[str, int] = {
    # -- bat tracking：2023 季中上線 --------------------------------------
    # 實測 2023-06-15 為 0%、2023-08-15 為 46.3%，故 2023 只有後半季有值。
    # 2024 起全季完整 (44.7%，約等於揮棒率，因為沒揮棒的球本來就沒有值)。
    # 影響：「上一季平均揮棒速度」「上一季快速揮棒比例」這兩個特徵，
    #       最早只能用在 2024 賽季 (以 2023 後半季為基準，且樣本偏頗)，
    #       真正乾淨的用法是 2025 起 (以完整的 2024 為基準)。
    "bat_speed": 2024,
    "swing_length": 2024,
    # -- 揮棒路徑細節：2024 起 --------------------------------------------
    "attack_angle": 2024,
    "attack_direction": 2024,
    "swing_path_tilt": 2024,
    "intercept_ball_minus_batter_pos_x_inches": 2024,
    "intercept_ball_minus_batter_pos_y_inches": 2024,
    # -- 手臂角度：2020 部分 (59%)，2021 起穩定 (97%+) ---------------------
    # 實測 2015-2019 皆為 0%。
    "arm_angle": 2021,
}

# 全程可用 (2015 起即有，抽樣皆 >90%) 的關鍵欄位。
# 列出來是為了明確：序列特徵與情境特徵不受年份限制，12 季都能用。
FULL_HISTORY_COLUMNS: tuple[str, ...] = (
    "plate_x", "plate_z", "release_speed", "effective_speed", "release_extension",
    "pfx_x", "pfx_z", "api_break_x_arm", "api_break_z_with_gravity",
    "release_spin_rate", "spin_axis", "release_pos_x", "release_pos_z",
    "pitch_type", "zone", "sz_top", "sz_bot",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "n_thruorder_pitcher", "pitcher_days_since_prev_game",
    "description", "events", "launch_speed", "launch_angle", "hc_x", "hc_y",
    "delta_run_exp", "estimated_woba_using_speedangle",
)

# 由上表推導：哪一年之後，全部規格欄位都齊備。
# 需要揮棒機制特徵的分析，建模賽季不應早於此。
FIRST_FULL_FEATURE_YEAR = 2025


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
        "base_state": (
            "壘包狀態。推導：on_1b/on_2b/on_3b 是否為 NaN -> 3 bit -> 8 個類別 "
            "(空壘/一壘/二壘/.../滿壘)。當類別用，不要用跑者人數，因為"
            "二三壘有人與一二壘有人對配球的意義不同 (前者不怕保送)。"
        ),
        "n_runners": "壘上人數 0-3。base_state 的粗粒度版，樣本不足時的退路。",
        "risp": "得點圈有人 (二壘或三壘)。投手在此情境會明顯改變配球。",
        "outs_when_up": (
            "出局數 0-2。已是 Statcast 原始欄位，不需推導，"
            "但必須與 base_state 一起看 (合稱 base-out state，24 種)。"
        ),
        "base_out_state": "base_state x outs 的 24 格狀態，得分期望值的標準座標",
    },
    # -- 打線脈絡：前幾棒發生了什麼 ----------------------------------------
    # 與 sequence 的差別：sequence 的單位是「球」(同一打席內)，
    # 這一組的單位是「打席」(同一場、同一進攻方、時間上更早的打席)。
    # 假設：投手/捕手會因為前幾棒被打爆或抓到節奏而調整配球。
    "lineup_context": {
        "prev{j}_pa_events": (
            "同一進攻方前 j 個打席的終結事件 (j=1,2,3)。推導：依 "
            "(game_pk, inning_topbot) 取 at_bat_number 排序，對 events 非空的列做 shift(j)。"
            "務必只用嚴格更早的打席，不可用到本打席自己的 events (那是未來資訊)。"
        ),
        "prev{j}_pa_result_class": (
            "把 events 收斂成 out / single / xbh / bb_hbp / k 五類。"
            "原始 events 有 20 種以上，直接當類別會樣本過薄。"
        ),
        "prev{j}_pa_launch_speed": "前 j 打席最後一球的擊球初速 (無觸球則缺值)",
        "prev_pa_reached_count": (
            "前 3 個打席中有幾個上壘。投手連續被上壘時傾向轉為保守配球。"
        ),
        "inning_batters_faced": (
            "本半局至今已面對的打者數。推導：(game_pk, inning, inning_topbot) 內"
            "相異 at_bat_number 的累計。與 base_out_state 一起描述「這局崩到什麼程度」。"
        ),
        "LEAKAGE_NOTE": (
            "本組全部特徵都必須嚴格使用「本打席開始之前」已完成的打席。"
            "用 shift 而非 rolling，且 shift 前務必先排序，否則會混入同打席或未來打席。"
        ),
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
        "plate_dx_prev1": "進壘水平位置差 (正規化後)，保留方向性，與距離互補",
        "plate_dz_prev1": "進壘垂直位置差 (正規化後)，保留方向性",
        "pfx_dx_prev1": "水平位移差 pfx_x - prev1_pfx_x。位移差與球速差是兩件事：\n"
                        "同樣 85 mph，橫move 差 10 吋的滑球與變速球，打者反應完全不同。",
        "pfx_dz_prev1": "垂直位移差 pfx_z - prev1_pfx_z",
        "pfx_dist_prev1": "位移向量的歐氏距離 sqrt(pfx_dx^2 + pfx_dz^2)，位移差的純量版",
        "same_family_prev1": "是否與前一球同球種族 (球種差的二元版)",
        "family_pair_prev1": "(prev1_pitch_family, pitch_family) 的有序配對，\n"
                             "球種差的類別版。序列統計的最小單位，比二元的 same/diff 保留更多資訊。",
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
    # -- 打者輪廓：一律用「上一季」或「近期」，不可用當季 -------------------
    # 這是整份 spec 最容易出錯的地方。打者輪廓若用當季資料計算，
    # 再拿去預測當季的每一球，等於用結果預測結果 —— 模型會虛高，結論無效。
    #
    # 兩種時間基準，用途不同：
    #   PREV_SEASON  上一季全季彙總。穩定、樣本大，但反應不了狀態變化。
    #                用於「這名打者本質上是什麼型」的特徵。
    #   ROLLING      本季截至前一天的滾動窗。會反應狀態，但早季樣本薄。
    #                用於「他最近手感如何」的特徵。
    "batter_profile": {
        "TIME_BASIS_NOTE": (
            "prev_season_* 以打者上一季 (game_year - 1) 全季計算；"
            "recent_* 以本季截至「本場比賽前一天」的滾動窗計算，窗長見 ROLLING_WINDOW。"
            "兩者都不得包含本場及之後的任何一球。"
        ),
        "ROLLING_WINDOW": "近期特徵用最近 30 天，且要求窗內至少 50 次揮棒，否則退回上一季值",

        # -- 揮棒機制 (需 bat tracking，2024 起才完整，見 COLUMN_AVAILABILITY) --
        "prev_season_mean_bat_speed": (
            "上一季平均揮棒速度 (mph)。只在有揮棒的球上平均。"
            "用上一季而非當季：揮棒速度是打者的體能特質，季內變化慢，"
            "用上一季既避免洩漏又幾乎不損失資訊。"
        ),
        "prev_season_fast_swing_rate": (
            "上一季快速揮棒比例。定義：bat_speed >= 75 mph 的揮棒佔全部揮棒的比例。"
            "75 mph 是 Statcast 官方 fast-swing 門檻。"
            "這比平均棒速更能分辨「always 全力揮」與「看球種調整」兩種打者。"
        ),
        "prev_season_mean_swing_length": "上一季平均揮棒軌跡長度 (ft)",
        "prev_season_mean_attack_angle": "上一季平均攻擊角 (度)，2024 起可用",

        # -- 拉打傾向 (用近期，因為打者會在季中調整打擊策略) -------------------
        "recent_pull_rate": (
            "近期拉打率。噴射角 = degrees(arctan2(hc_x-125.42, 198.27-hc_y))，"
            "右打者 < -15 度為拉打，左打者 > 15 度為拉打。"
            "改用近期而非上一季：拉打傾向是可調整的策略 (打者會因應佈陣改變)，"
            "季內變化比揮棒速度大得多。"
            "不用 FanGraphs Pull%，因為該來源目前回 403。"
        ),
        "recent_pull_rate_by_family": "各球種族的近期拉打率，樣本不足時收縮回整體值",

        # -- 反應輪廓：分群的主要依據 ------------------------------------------
        # 分群特徵必須是「反應」不是「結果」。用打擊率只會分出強打者與弱打者。
        "prev_season_swing_rate_by_family_zone": "上一季各球種族 x 好球帶區塊的揮棒率",
        "prev_season_whiff_rate_by_family_zone": "上一季各球種族 x 區塊的揮空率",
        "prev_season_chase_rate_by_family": "上一季各球種族的追打率 (好球帶外出棒)",
        "prev_season_ev_by_family": "上一季各球種族的平均擊球初速",

        "SHRINKAGE_NOTE": (
            "所有比率特徵必須做 shrinkage 往聯盟平均收縮，權重依樣本數。"
            "某打者在某區塊只遇過 5 球就決定他的分群，比不做標準化還危險。"
        ),
        "COLD_START_NOTE": (
            "新人與上一季未出賽者沒有 prev_season_* 值。不可填 0 (那代表「揮棒速度 0」)。"
            "做法：填聯盟平均並另開一個 is_rookie 指示欄位，讓模型自己決定怎麼用。"
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
