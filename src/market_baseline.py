"""
博彩市场基线模型
===============
从博彩赔率提取市场隐含概率，作为基线参考和集成输入。

原理
----
赔率隐含概率 = 1 / 赔率
去除博彩公司毛利（overround）后得到市场共识概率。
多家公司取均值可减少单一公司偏差。

用途
----
1. 作为第四个"基模型"输入 Stacking 集成
2. 作为回测基准——如果统计模型持续被市场打败，需要反思
3. 检测模型与市场的显著偏离（可能的机会或漏洞）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize


class MarketBaseline:
    """
    博彩市场隐含概率提取。

    支持的赔率格式
    ------------
    - 十进制 (Decimal): 2.50, 3.20, 2.80
    - 默认假设输入为十进制赔率

    参数
    ----------
    margin_method : str
        毛利去除方法:
        - "proportional": 按比例分摊（最常用）
        - "additive": 加法分摊
        - "shin": Shin 模型（考虑信息不对称，更准确）
    """

    def __init__(self, margin_method: str = "shin"):
        self.margin_method = margin_method
        self._fitted = False

    def implied_probabilities(
        self,
        odds_home: np.ndarray,
        odds_draw: np.ndarray,
        odds_away: np.ndarray,
    ) -> np.ndarray:
        """
        从赔率提取隐含概率。

        参数
        ----------
        odds_home : np.ndarray
            主胜赔率
        odds_draw : np.ndarray
            平局赔率
        odds_away : np.ndarray
            客胜赔率

        返回
        -------
        np.ndarray
            形状 (n, 3)，每列依次为 [主胜, 平局, 客胜] 概率，和为 1
        """
        if self.margin_method == "proportional":
            return self._proportional_margin(odds_home, odds_draw, odds_away)
        elif self.margin_method == "additive":
            return self._additive_margin(odds_home, odds_draw, odds_away)
        elif self.margin_method == "shin":
            return self._shin_margin(odds_home, odds_draw, odds_away)
        else:
            raise ValueError(f"未知毛利去除方法: {self.margin_method}")

    @staticmethod
    def _proportional_margin(
        odds_home: np.ndarray,
        odds_draw: np.ndarray,
        odds_away: np.ndarray,
    ) -> np.ndarray:
        """
        按比例分摊毛利。

        overround = 1/o1 + 1/o2 + 1/o3
        原始概率 = (1/oi) / overround
        """
        raw_h = 1.0 / odds_home
        raw_d = 1.0 / odds_draw
        raw_a = 1.0 / odds_away
        overround = raw_h + raw_d + raw_a
        return np.column_stack([raw_h / overround, raw_d / overround, raw_a / overround])

    @staticmethod
    def _additive_margin(
        odds_home: np.ndarray,
        odds_draw: np.ndarray,
        odds_away: np.ndarray,
    ) -> np.ndarray:
        """
        加法分摊。

        从每个原始概率中减去 overround / 3。
        """
        n = len(odds_home)
        raw_h = 1.0 / odds_home
        raw_d = 1.0 / odds_draw
        raw_a = 1.0 / odds_away
        overround = raw_h + raw_d + raw_a
        margin_per_outcome = (overround - 1.0) / 3.0

        prob_h = np.maximum(raw_h - margin_per_outcome, 0.001)
        prob_d = np.maximum(raw_d - margin_per_outcome, 0.001)
        prob_a = np.maximum(raw_a - margin_per_outcome, 0.001)

        total = prob_h + prob_d + prob_a
        return np.column_stack([prob_h / total, prob_d / total, prob_a / total])

    @staticmethod
    def _shin_margin(
        odds_home: np.ndarray,
        odds_draw: np.ndarray,
        odds_away: np.ndarray,
    ) -> np.ndarray:
        """
        Shin (1993) 模型——考虑信息不对称。

        假设部分资金来自内幕信息，通过数值优化求解 z 参数。
        这是学术文献中最常推荐的隐式概率提取方法。
        """
        n = len(odds_home)
        probs = np.zeros((n, 3))

        for i in range(n):
            o = np.array([odds_home[i], odds_draw[i], odds_away[i]])
            raw = 1.0 / o

            # 二分法求解 Shin 参数 z
            def objective(z):
                z = z[0]
                if z <= 0 or z >= 1:
                    return 1e10
                denom = 3.0
                sum_sqrt = np.sum(np.sqrt(z**2 + 4 * (1 - z) * raw**2 / denom))
                return (sum_sqrt - 2)**2

            result = minimize(objective, x0=[0.01], bounds=[(0.001, 0.5)], method="L-BFGS-B")
            z = max(0.001, min(0.5, result.x[0]))

            # 计算真实概率
            p = np.zeros(3)
            denom = 3.0
            for j in range(3):
                p[j] = (np.sqrt(z**2 + 4 * (1 - z) * raw[j]**2 / denom) - z) / (2 * (1 - z))
            p /= p.sum()
            probs[i] = p

        return probs

    # ═══════════════════════════════════════
    # 预测接口
    # ═══════════════════════════════════════

    def predict_single(
        self, odds_home: float, odds_draw: float, odds_away: float
    ) -> Dict[str, float]:
        """
        从单场赔率预测概率。

        参数
        ----------
        odds_home : float
            主胜赔率
        odds_draw : float
            平局赔率
        odds_away : float
            客胜赔率

        返回
        -------
        Dict
            {"home_win_pct", "draw_pct", "away_win_pct", "overround_pct"}
        """
        probs = self.implied_probabilities(
            np.array([odds_home]),
            np.array([odds_draw]),
            np.array([odds_away]),
        )[0]

        overround = (1 / odds_home + 1 / odds_draw + 1 / odds_away - 1) * 100

        return {
            "home_win_pct": round(probs[0] * 100, 1),
            "draw_pct": round(probs[1] * 100, 1),
            "away_win_pct": round(probs[2] * 100, 1),
            "overround_pct": round(overround, 1),
        }

    def predict_df(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        从赔率 DataFrame 批量计算市场隐含概率。

        参数
        ----------
        odds_df : pd.DataFrame
            含 home_odds, draw_odds, away_odds 列

        返回
        -------
        pd.DataFrame
            含 market_home_pct, market_draw_pct, market_away_pct, market_overround
        """
        probs = self.implied_probabilities(
            odds_df["home_odds"].values,
            odds_df["draw_odds"].values,
            odds_df["away_odds"].values,
        )
        result = odds_df.copy()
        result["market_home_pct"] = (probs[:, 0] * 100).round(1)
        result["market_draw_pct"] = (probs[:, 1] * 100).round(1)
        result["market_away_pct"] = (probs[:, 2] * 100).round(1)
        result["market_overround"] = (
            (1 / odds_df["home_odds"] + 1 / odds_df["draw_odds"] + 1 / odds_df["away_odds"] - 1) * 100
        ).round(1)
        return result


# ═══════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════

def odds_to_model_format(
    home_odds: float, draw_odds: float, away_odds: float
) -> np.ndarray:
    """
    将赔率转换为模型统一概率格式 [客胜, 平局, 主胜]。

    参数
    ----------
    home_odds : float
        主胜赔率
    draw_odds : float
        平局赔率
    away_odds : float
        客胜赔率
    """
    market = MarketBaseline(margin_method="shin")
    probs = market.implied_probabilities(
        np.array([home_odds]),
        np.array([draw_odds]),
        np.array([away_odds]),
    )[0]
    # 重排为 [客胜, 平局, 主胜] 以匹配其他模型
    return np.array([probs[2], probs[1], probs[0]])


if __name__ == "__main__":
    print("博彩市场基线模块加载成功。")

    # 简单测试
    market = MarketBaseline()
    result = market.predict_single(2.50, 3.20, 2.80)
    print(f"赔率 (2.50, 3.20, 2.80) → 市场概率: {result}")
