"""
Elo 评分系统
===========
基于 Elo 算法的球队实力评分，作为多模型集成中的基准线。

增强（相比基础 Elo）：
1. K值按赛事重要性分级
2. 净胜球加成（不再只看胜负）
3. 主场/中立场地修正
4. 回归均值因子（长时间不比赛的球队向均值回归）
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class EloRating:
    """
    Elo 评分系统。

    参数
    ----------
    initial_rating : float
        新球队的默认初始评分
    k_factors : Dict[str, float]
        各赛事类型的 K 值
    regression_mean : float
        回归均值目标
    regression_rate : float
        每30天向均值回归的比例
    home_advantage : float
        主场优势 Elo 加成（约100分）
    """

    def __init__(
        self,
        initial_rating: float = 1300.0,
        k_factors: Optional[Dict[str, float]] = None,
        regression_mean: float = 1300.0,
        regression_rate: float = 0.02,  # 每30天
        home_advantage: float = 100.0,
    ):
        self.ratings: Dict[str, float] = {}
        self.last_match_date: Dict[str, datetime] = {}
        self.initial_rating = initial_rating
        self.k_factors = k_factors or {
            "World Cup": 60,
            "Continental Cup": 40,
            "Qualifier": 30,
            "Confederations Cup": 35,
            "Nations League": 25,
            "Friendly": 20,
        }
        self.regression_mean = regression_mean
        self.regression_rate = regression_rate
        self.home_advantage = home_advantage
        self.match_count: Dict[str, int] = {}

    # ═══════════════════════════════════════
    # 核心方法
    # ═══════════════════════════════════════

    def expected_score(
        self, rating_a: float, rating_b: float, is_neutral: bool = False
    ) -> float:
        """
        计算 A 队的预期得分（0~1）。

        参数
        ----------
        rating_a : float
            A队评分（含可能的主动场加成）
        rating_b : float
            B队评分
        is_neutral : bool
            中立场地则不加主场优势

        返回
        -------
        float
            A 队预期胜率
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: int,
        score_b: int,
        tournament: str,
        match_date: Optional[datetime] = None,
        is_neutral: bool = False,
    ) -> Tuple[float, float]:
        """
        根据比赛结果更新两队 Elo 评分。

        参数
        ----------
        team_a, team_b : str
            两队名称
        score_a, score_b : int
            进球数
        tournament : str
            赛事类型
        match_date : datetime or None
            比赛日期（用于回归均值计算）
        is_neutral : bool
            中立场地

        返回
        -------
        Tuple[float, float]
            (team_a 评分变化量, team_b 评分变化量)
        """
        # 初始化新球队
        for team in [team_a, team_b]:
            if team not in self.ratings:
                self.ratings[team] = self.initial_rating
                self.match_count[team] = 0

        # 回归均值（长时间不比赛）
        if match_date is not None:
            for team in [team_a, team_b]:
                if team in self.last_match_date:
                    days_gap = (match_date - self.last_match_date[team]).days
                    if days_gap > 30:
                        n_periods = days_gap / 30.0
                        for _ in range(int(n_periods)):
                            self.ratings[team] += self.regression_rate * (
                                self.regression_mean - self.ratings[team]
                            )

        # 记录比赛日
        if match_date is not None:
            self.last_match_date[team_a] = match_date
            self.last_match_date[team_b] = match_date

        # K 值
        k = self.k_factors.get(tournament, 20)

        # 净胜球加成
        goal_diff = abs(score_a - score_b)
        if goal_diff == 0:
            goal_bonus = 1.0
        elif goal_diff == 1:
            goal_bonus = 1.0
        elif goal_diff == 2:
            goal_bonus = 1.3
        elif goal_diff == 3:
            goal_bonus = 1.5
        else:
            goal_bonus = 1.75

        # 有效 K 值
        k_eff = k * goal_bonus

        # 实际结果
        if score_a > score_b:
            actual_a, actual_b = 1.0, 0.0
        elif score_a < score_b:
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a, actual_b = 0.5, 0.5

        # 主场加成后的评分
        rating_a_adj = self.ratings[team_a] + (0 if is_neutral else self.home_advantage)
        rating_b_adj = self.ratings[team_b]

        # 预期
        expected_a = self.expected_score(rating_a_adj, rating_b_adj)
        expected_b = 1.0 - expected_a

        # 更新
        delta_a = k_eff * (actual_a - expected_a)
        delta_b = k_eff * (actual_b - expected_b)

        self.ratings[team_a] += delta_a
        self.ratings[team_b] += delta_b

        self.match_count[team_a] = self.match_count.get(team_a, 0) + 1
        self.match_count[team_b] = self.match_count.get(team_b, 0) + 1

        return delta_a, delta_b

    # ═══════════════════════════════════════
    # 预测
    # ═══════════════════════════════════════

    def predict(
        self,
        team_a: str,
        team_b: str,
        is_neutral: bool = False,
        with_draw: bool = True,
    ) -> Dict[str, float]:
        """
        预测单场比赛结果概率。

        参数
        ----------
        team_a, team_b : str
            两队名称
        is_neutral : bool
            中立场地
        with_draw : bool
            是否估算平局概率

        返回
        -------
        Dict
            {"home_win_pct", "draw_pct", "away_win_pct", "elo_diff"}
        """
        r_a = self.ratings.get(team_a, self.initial_rating)
        r_b = self.ratings.get(team_b, self.initial_rating)

        r_a_adj = r_a + (0 if is_neutral else self.home_advantage)
        expected_a = self.expected_score(r_a_adj, r_b)

        if with_draw:
            # 用 Elo 分差估算平局概率
            # 分差越小，平局概率越高
            elo_diff = abs(r_a_adj - r_b)
            # 经验公式：平局概率 ≈ 0.32 * exp(-elo_diff / 400)
            draw_prob = 0.32 * np.exp(-elo_diff / 400)
            home_win = expected_a - draw_prob / 2
            away_win = (1 - expected_a) - draw_prob / 2

            # 确保非负
            home_win = max(0.01, home_win)
            away_win = max(0.01, away_win)
            draw_prob = max(0.01, draw_prob)

            # 归一化
            total = home_win + draw_prob + away_win
            home_win /= total
            draw_prob /= total
            away_win /= total
        else:
            home_win = expected_a
            away_win = 1 - expected_a
            draw_prob = 0.0

        return {
            "home_win_pct": round(home_win * 100, 1),
            "draw_pct": round(draw_prob * 100, 1),
            "away_win_pct": round(away_win * 100, 1),
            "elo_diff": round(r_a - r_b, 1),
            "elo_home": round(r_a, 1),
            "elo_away": round(r_b, 1),
        }

    # ═══════════════════════════════════════
    # 批量训练和导出
    # ═══════════════════════════════════════

    def fit_all(
        self,
        matches: pd.DataFrame,
        home_col: str = "home_team",
        away_col: str = "away_team",
        goals_h_col: str = "home_goals",
        goals_a_col: str = "away_goals",
        tournament_col: str = "tournament_type",
        date_col: str = "date",
        neutral_col: Optional[str] = "neutral",
    ) -> "EloRating":
        """
        按时间顺序批量处理所有历史比赛。

        参数
        ----------
        matches : pd.DataFrame
            历史比赛数据
        """
        matches = matches.sort_values(date_col).copy()

        for _, row in matches.iterrows():
            is_neutral = False
            if neutral_col and neutral_col in matches.columns:
                is_neutral = row[neutral_col] in ("Y", "y", 1, True, "1")

            self.update(
                team_a=row[home_col],
                team_b=row[away_col],
                score_a=int(row[goals_h_col]),
                score_b=int(row[goals_a_col]),
                tournament=row.get(tournament_col, "Friendly"),
                match_date=row[date_col],
                is_neutral=is_neutral,
            )

        return self

    def get_ratings_df(self) -> pd.DataFrame:
        """返回评分 DataFrame。"""
        rows = []
        for team, rating in self.ratings.items():
            rows.append(
                {
                    "team": team,
                    "elo_rating": round(rating, 1),
                    "matches_played": self.match_count.get(team, 0),
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values("elo_rating", ascending=False).reset_index(drop=True)

    def get_ratings_dict(self) -> Dict[str, float]:
        """返回评分字典。"""
        return dict(self.ratings)


if __name__ == "__main__":
    print("Elo 评分模块加载成功。")
    print("使用方法: elo = EloRating(); elo.update(...)")
