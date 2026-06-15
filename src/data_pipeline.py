"""
数据清洗与特征工程流水线
=======================
负责：时间加权、缺失值处理、特征构建、数据质量检查
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════

# 赛事权重配置
TOURNAMENT_WEIGHTS = {
    "World Cup": 3.0,
    "Continental Cup": 2.0,
    "Qualifier": 1.5,
    "Confederations Cup": 1.5,
    "Friendly": 0.8,
    "Nations League": 1.3,  # 欧国联等
}

# 洲际杯赛事映射（自动识别）
CONTINENTAL_CUPS = [
    "Copa America", "Euro", "AFC Asian Cup", "Africa Cup of Nations",
    "Gold Cup", "OFC Nations Cup", "Asian Cup", "African Cup"
]


def classify_tournament(name: str) -> str:
    """
    依据赛事名称自动分类。

    参数
    ----------
    name : str
        赛事名称

    返回
    -------
    str
        赛事类别：World Cup / Continental Cup / Qualifier / Nations League / Friendly
    """
    name_lower = name.lower()

    if "world cup" in name_lower and "qualif" not in name_lower:
        return "World Cup"

    for cup in CONTINENTAL_CUPS:
        if cup.lower() in name_lower:
            return "Continental Cup"

    if "qualif" in name_lower or "qualify" in name_lower:
        return "Qualifier"

    if "nations league" in name_lower:
        return "Nations League"

    if "confederations cup" in name_lower:
        return "Confederations Cup"

    return "Friendly"


# ═══════════════════════════════════════════
# 时间加权
# ═══════════════════════════════════════════

def calculate_match_weight(
    match_date: datetime,
    reference_date: datetime,
    tournament: str,
    is_knockout: bool = False,
    half_life_days: int = 730
) -> float:
    """
    计算单场比赛的样本权重。

    核心思想：越近的比赛、越重要的赛事、淘汰赛阶段——权重越高。

    参数
    ----------
    match_date : datetime
        比赛日期
    reference_date : datetime
        参考日期（预测日/回测基准日）
    tournament : str
        赛事类型
    is_knockout : bool
        是否为淘汰赛
    half_life_days : int
        时间衰减半衰期（天），默认2年

    返回
    -------
    float
        归一化后的样本权重
    """
    days_ago = (reference_date - match_date).days

    # 不允许未来日期参与训练
    if days_ago < 0:
        return 0.0

    # 时间衰减：指数衰减
    time_weight = 0.5 ** (days_ago / half_life_days)

    # 赛事类型加成
    tournament_mult = TOURNAMENT_WEIGHTS.get(tournament, 1.0)

    # 淘汰赛额外加成
    knockout_bonus = 1.3 if is_knockout else 1.0

    return time_weight * tournament_mult * knockout_bonus


# ═══════════════════════════════════════════
# 近期状态特征
# ═══════════════════════════════════════════

def compute_recent_form(
    matches: pd.DataFrame,
    team_col: str = "team",
    date_col: str = "date",
    goals_for_col: str = "goals_for",
    goals_against_col: str = "goals_against",
    reference_date: Optional[datetime] = None,
    window_matches: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    计算每支球队近期状态指标。

    参数
    ----------
    matches : pd.DataFrame
        历史比赛数据（每行是某队在某场比赛中的表现）
    reference_date : datetime or None
        参考日期，None 则用最新日期
    window_matches : List[int]
        滑动窗口大小（按比赛场次）

    返回
    -------
    pd.DataFrame
        每队的状态指标：胜率、场均进球、场均失球、净胜球
    """
    if reference_date is None:
        reference_date = matches[date_col].max()

    results = []

    for team in matches[team_col].unique():
        team_matches = matches[matches[team_col] == team].copy()
        team_matches = team_matches[team_matches[date_col] <= reference_date]
        team_matches = team_matches.sort_values(date_col, ascending=False)

        row = {"team": team}

        for w in window_matches:
            recent = team_matches.head(w)
            if len(recent) < w:
                row[f"form_{w}_win_rate"] = np.nan
                row[f"form_{w}_gf_avg"] = np.nan
                row[f"form_{w}_ga_avg"] = np.nan
                row[f"form_{w}_gd_avg"] = np.nan
            else:
                wins = (recent[goals_for_col] > recent[goals_against_col]).sum()
                row[f"form_{w}_win_rate"] = wins / w
                row[f"form_{w}_gf_avg"] = recent[goals_for_col].mean()
                row[f"form_{w}_ga_avg"] = recent[goals_against_col].mean()
                row[f"form_{w}_gd_avg"] = (recent[goals_for_col] - recent[goals_against_col]).mean()

        results.append(row)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════
# 比赛级别特征构建
# ═══════════════════════════════════════════

def build_match_features(
    home_team: str,
    away_team: str,
    match_date: datetime,
    team_params: pd.DataFrame,
    elo_ratings: Dict[str, float],
    recent_form: pd.DataFrame,
    team_attributes: pd.DataFrame,
    h2h_history: Optional[pd.DataFrame] = None,
    rest_days: Optional[Dict[str, int]] = None,
    venue_altitude: float = 0.0,
    is_neutral: bool = False,
) -> Dict[str, float]:
    """
    为单场比赛构建完整特征向量。

    参数
    ----------
    home_team, away_team : str
        主客队名称
    match_date : datetime
        比赛日期
    team_params : pd.DataFrame
        泊松模型输出的攻击/防守参数
    elo_ratings : Dict[str, float]
        当前 Elo 评分
    recent_form : pd.DataFrame
        近期状态数据
    team_attributes : pd.DataFrame
        球队属性（FIFA排名、身价等）
    h2h_history : pd.DataFrame or None
        历史交锋记录
    rest_days : Dict[str, int] or None
        各队休息天数
    venue_altitude : float
        比赛场地海拔（米）
    is_neutral : bool
        是否中立场地

    返回
    -------
    Dict[str, float]
        特征字典
    """
    features = {}

    # --- 攻防参数差 ---
    home_attack = team_params.loc[team_params["team"] == home_team, "attack_strength"].values
    home_defense = team_params.loc[team_params["team"] == home_team, "defense_strength"].values
    away_attack = team_params.loc[team_params["team"] == away_team, "attack_strength"].values
    away_defense = team_params.loc[team_params["team"] == away_team, "defense_strength"].values

    features["attack_diff"] = (float(home_attack[0]) - float(away_attack[0])) if len(home_attack) and len(away_attack) else 0.0
    features["defense_diff"] = (float(home_defense[0]) - float(away_defense[0])) if len(home_defense) and len(away_defense) else 0.0
    features["overall_rating_diff"] = features["attack_diff"] - features["defense_diff"]

    # --- Elo 差 ---
    elo_h = elo_ratings.get(home_team, 1300)
    elo_a = elo_ratings.get(away_team, 1300)
    features["elo_diff"] = elo_h - elo_a

    # --- 近期状态差 ---
    h_form = recent_form[recent_form["team"] == home_team]
    a_form = recent_form[recent_form["team"] == away_team]

    for w in [5, 10]:
        for metric in ["win_rate", "gf_avg", "ga_avg", "gd_avg"]:
            col = f"form_{w}_{metric}"
            h_val = float(h_form[col].values[0]) if len(h_form) and not pd.isna(h_form[col].values[0]) else 0.0
            a_val = float(a_form[col].values[0]) if len(a_form) and not pd.isna(a_form[col].values[0]) else 0.0
            features[f"form_{w}_{metric}_diff"] = h_val - a_val

    # --- 球队属性差 ---
    for attr_col in ["fifa_rank", "squad_value_billion", "avg_age", "world_cup_appearances"]:
        if attr_col in team_attributes.columns:
            h_val = team_attributes.loc[team_attributes["team"] == home_team, attr_col].values
            a_val = team_attributes.loc[team_attributes["team"] == away_team, attr_col].values
            if len(h_val) and len(a_val):
                if attr_col == "fifa_rank":
                    # 排名越低越好，所以反过来减
                    features["rank_diff"] = float(a_val[0]) - float(h_val[0])
                elif attr_col == "squad_value_billion":
                    features["value_diff_log"] = np.log1p(float(h_val[0])) - np.log1p(float(a_val[0]))
                else:
                    features[f"{attr_col}_diff"] = float(h_val[0]) - float(a_val[0])

    # --- 休息天数差 ---
    if rest_days:
        features["rest_days_diff"] = rest_days.get(home_team, 5) - rest_days.get(away_team, 5)
    else:
        features["rest_days_diff"] = 0.0

    # --- 海拔因子 ---
    features["venue_altitude"] = venue_altitude

    # --- 中立场地 ---
    features["is_neutral"] = 1.0 if is_neutral else 0.0

    # --- 历史交锋优势 ---
    if h2h_history is not None and len(h2h_history) > 0:
        h2h_wins = len(h2h_history[(h2h_history["home_team"] == home_team) & (h2h_history["home_goals"] > h2h_history["away_goals"])])
        h2h_wins += len(h2h_history[(h2h_history["away_team"] == home_team) & (h2h_history["away_goals"] > h2h_history["home_goals"])])
        h2h_total = len(h2h_history)
        features["h2h_advantage"] = h2h_wins / h2h_total if h2h_total > 0 else 0.5
    else:
        features["h2h_advantage"] = 0.5

    return features


# ═══════════════════════════════════════════
# 数据质量检查
# ═══════════════════════════════════════════

def check_data_quality(
    fixtures: pd.DataFrame,
    match_history: pd.DataFrame,
    team_attributes: pd.DataFrame
) -> List[str]:
    """
    检查数据完整性和质量。

    返回
    -------
    List[str]
        数据质量问题列表（空列表表示数据质量通过）
    """
    issues = []

    # 1. 检查48支参赛队是否都有历史数据
    required_teams = set(fixtures["home_team"].unique()) | set(fixtures["away_team"].unique())
    history_teams = set(match_history["home_team"].unique()) | set(match_history["away_team"].unique())
    attr_teams = set(team_attributes["team"].unique())

    missing_history = required_teams - history_teams
    missing_attr = required_teams - attr_teams

    if missing_history:
        issues.append(f"⚠️ {len(missing_history)} 支球队缺少历史比赛数据: {', '.join(sorted(missing_history))}")

    if missing_attr:
        issues.append(f"⚠️ {len(missing_attr)} 支球队缺少属性数据: {', '.join(sorted(missing_attr))}")

    # 2. 检查历史比赛场次是否足够
    for team in required_teams:
        count = len(match_history[
            (match_history["home_team"] == team) | (match_history["away_team"] == team)
        ])
        if count < 20:
            issues.append(f"⚠️ {team} 历史比赛不足20场（仅{count}场），预测置信度降低")

    # 3. 检查是否有未来日期的比赛
    today = datetime.now()
    future_matches = match_history[match_history["date"] > today]
    if len(future_matches):
        issues.append(f"⚠️ 历史数据中包含 {len(future_matches)} 场未来日期的比赛，可能数据有误")

    return issues


# ═══════════════════════════════════════════
# 主流水线
# ═══════════════════════════════════════════

def run_data_pipeline(
    match_history_path: str,
    fixtures_path: str = None,
    team_attributes_path: str = None,
    reference_date: Optional[datetime] = None,
    half_life_days: int = 730,
) -> Dict:
    """
    执行完整的数据预处理流水线。

    参数
    ----------
    match_history_path : str
        历史比赛 CSV 路径
    fixtures_path : str or None
        赛程 CSV 路径
    team_attributes_path : str or None
        球队属性 CSV 路径
    reference_date : datetime or None
        基准日期
    half_life_days : int
        权重衰减半衰期

    返回
    -------
    Dict
        包含清洗后数据的字典
    """
    # 加载数据
    matches = pd.read_csv(match_history_path)
    matches["date"] = pd.to_datetime(matches["date"])

    if reference_date is None:
        reference_date = matches["date"].max()

    print(f"[数据流水线] 基准日期: {reference_date.date()}")
    print(f"[数据流水线] 加载 {len(matches)} 场历史比赛")

    # 分类赛事
    if "tournament_type" not in matches.columns:
        matches["tournament_type"] = matches["tournament"].apply(classify_tournament)

    # 检测淘汰赛
    if "is_knockout" not in matches.columns:
        matches["is_knockout"] = matches.get("stage", "").str.contains(
            "final|semi|quarter|round of|knockout|淘汰", case=False, na=False
        )

    # 计算权重
    matches["weight"] = matches.apply(
        lambda r: calculate_match_weight(
            r["date"], reference_date, r["tournament_type"], r["is_knockout"], half_life_days
        ),
        axis=1,
    )

    # 过滤未来数据和零权重
    matches = matches[matches["weight"] > 0].copy()

    # 加载赛程和属性
    fixtures = pd.read_csv(fixtures_path) if fixtures_path else None
    team_attributes = pd.read_csv(team_attributes_path) if team_attributes_path else None

    # 质量检查
    if fixtures is not None and team_attributes is not None:
        issues = check_data_quality(fixtures, matches, team_attributes)
        for issue in issues:
            print(f"  {issue}")
        if not issues:
            print(f"  ✅ 数据质量检查通过")

    print(f"[数据流水线] 有效比赛: {len(matches)}, 赛事类型: {matches['tournament_type'].nunique()}")
    print(f"[数据流水线] 覆盖球队: {matches['home_team'].nunique()}")

    return {
        "matches": matches,
        "fixtures": fixtures,
        "team_attributes": team_attributes,
        "reference_date": reference_date,
    }


if __name__ == "__main__":
    print("数据流水线模块加载成功。")
    print(f"支持赛事类型: {list(TOURNAMENT_WEIGHTS.keys())}")
