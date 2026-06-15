"""
单场比赛预测引擎
===============
整合四个模型（Dixon-Coles 泊松、Elo、XGBoost、博彩市场），
经由 Stacking 集成和概率校准，输出最终预测。

这是整个系统的核心预测入口。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import pickle

from poisson_model import DixonColesModel
from elo_calculator import EloRating
from xgboost_model import XGBoostMatchModel
from stacking_ensemble import StackingEnsemble
from market_baseline import MarketBaseline
from calibration import ProbabilityCalibrator
from data_pipeline import build_match_features


class MatchPredictor:
    """
    单场比赛预测器。

    参数
    ----------
    poisson_model : DixonColesModel
        泊松模型
    elo : EloRating
        Elo 评分系统
    xgb_model : XGBoostMatchModel
        XGBoost 模型
    stacking : StackingEnsemble
        Stacking 集成
    calibrator : ProbabilityCalibrator or None
        概率校准器（可选）
    market : MarketBaseline or None
        博彩市场基线（可选，有赔率时启用）
    team_attributes : pd.DataFrame or None
        球队属性
    recent_form : pd.DataFrame or None
        近期状态
    """

    def __init__(
        self,
        poisson_model: DixonColesModel,
        elo: EloRating,
        xgb_model: XGBoostMatchModel,
        stacking: StackingEnsemble,
        calibrator: Optional[ProbabilityCalibrator] = None,
        market: Optional[MarketBaseline] = None,
        team_attributes: Optional[pd.DataFrame] = None,
        recent_form: Optional[pd.DataFrame] = None,
    ):
        self.poisson = poisson_model
        self.elo = elo
        self.xgb = xgb_model
        self.stacking = stacking
        self.calibrator = calibrator
        self.market = market or MarketBaseline()
        self.team_attributes = team_attributes
        self.recent_form = recent_form

    def predict(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
        odds: Optional[Tuple[float, float, float]] = None,
        rest_days: Optional[Dict[str, int]] = None,
        venue_altitude: float = 0.0,
    ) -> Dict:
        """
        预测单场比赛结果。

        参数
        ----------
        home_team : str
            主队名称
        away_team : str
            客队名称
        is_neutral : bool
            是否为中立场地
        odds : Tuple[float, float, float] or None
            (主胜赔率, 平局赔率, 客胜赔率)，无赔率则跳过市场模型
        rest_days : Dict[str, int] or None
            各队休息天数 {"Brazil": 5, "Argentina": 4}
        venue_altitude : float
            比赛场地海拔（米）

        返回
        -------
        Dict
            {
                "home_win_pct": float,    最终主胜概率 %
                "draw_pct": float,        最终平局概率 %
                "away_win_pct": float,    最终客胜概率 %
                "most_likely_score": str, 最常见比分
                "expected_goals": (float, float),
                "confidence": str,        置信度 (高/中/低)
                "model_agreement": str,   模型一致性 (一致/分歧/严重分歧)
                "individual_models": Dict, 各基模型独立预测
                "features_used": Dict,    使用的关键特征
            }
        """
        # ── 1. 泊松模型预测 ──
        poisson_result = self.poisson.predict_result(home_team, away_team, is_neutral)

        # ── 2. Elo 模型预测 ──
        elo_result = self.elo.predict(home_team, away_team, is_neutral, with_draw=True)

        # ── 3. XGBoost 模型预测 ──
        # 构建特征
        features = build_match_features(
            home_team=home_team,
            away_team=away_team,
            match_date=pd.Timestamp.now(),  # 预测用当前日期
            team_params=self.poisson.get_team_params_df(),
            elo_ratings=self.elo.get_ratings_dict(),
            recent_form=self.recent_form if self.recent_form is not None else pd.DataFrame(),
            team_attributes=self.team_attributes if self.team_attributes is not None else pd.DataFrame(),
            rest_days=rest_days,
            venue_altitude=venue_altitude,
            is_neutral=is_neutral,
        )

        xgb_result = self.xgb.predict_single(features)

        # ── 4. 博彩市场（如有赔率）──
        market_result = None
        if odds:
            market_result = self.market.predict_single(*odds)

        # ── 5. 收集基模型预测 ──
        base_preds = {
            "poisson": np.array([
                poisson_result["away_win_pct"] / 100,
                poisson_result["draw_pct"] / 100,
                poisson_result["home_win_pct"] / 100,
            ]),
            "elo": np.array([
                elo_result["away_win_pct"] / 100,
                elo_result["draw_pct"] / 100,
                elo_result["home_win_pct"] / 100,
            ]),
            "xgboost": np.array([
                xgb_result["away_win_pct"] / 100,
                xgb_result["draw_pct"] / 100,
                xgb_result["home_win_pct"] / 100,
            ]),
        }

        if market_result:
            base_preds["market"] = np.array([
                market_result["away_win_pct"] / 100,
                market_result["draw_pct"] / 100,
                market_result["home_win_pct"] / 100,
            ])

        # ── 6. Stacking 集成 ──
        try:
            stacked_proba = self.stacking.predict_single(base_preds)
            # 转换为 np.array 用于校准
            proba_array = np.array([[
                stacked_proba["away_win_pct"] / 100,
                stacked_proba["draw_pct"] / 100,
                stacked_proba["home_win_pct"] / 100,
            ]])
        except Exception:
            # 回退到简单平均
            all_probs = np.array([v for v in base_preds.values()])
            avg = all_probs.mean(axis=0)
            proba_array = avg.reshape(1, -1)

        # ── 7. 校准 ──
        if self.calibrator and self.calibrator._fitted:
            proba_array = self.calibrator.predict_proba(proba_array)

        final_probs = proba_array[0]

        # ── 8. 置信度与一致性评估 ──
        max_prob = final_probs.max()
        if max_prob > 0.55:
            confidence = "高"
        elif max_prob > 0.40:
            confidence = "中"
        else:
            confidence = "低"

        # 模型间分歧
        model_home_probs = [p[2] for p in base_preds.values()]
        model_spread = max(model_home_probs) - min(model_home_probs)

        if model_spread < 0.05:
            agreement = "一致"
        elif model_spread < 0.15:
            agreement = "分歧"
        else:
            agreement = "严重分歧"

        # ── 9. 组装结果 ──
        individual = {
            "poisson": {
                "home_win": poisson_result["home_win_pct"],
                "draw": poisson_result["draw_pct"],
                "away_win": poisson_result["away_win_pct"],
                "expected_goals": (
                    poisson_result["expected_home_goals"],
                    poisson_result["expected_away_goals"],
                ),
                "most_likely_score": poisson_result["most_likely_score"],
            },
            "elo": {
                "home_win": elo_result["home_win_pct"],
                "draw": elo_result["draw_pct"],
                "away_win": elo_result["away_win_pct"],
                "elo_diff": elo_result["elo_diff"],
            },
            "xgboost": {
                "home_win": xgb_result["home_win_pct"],
                "draw": xgb_result["draw_pct"],
                "away_win": xgb_result["away_win_pct"],
            },
        }

        if market_result:
            individual["market"] = {
                "home_win": market_result["home_win_pct"],
                "draw": market_result["draw_pct"],
                "away_win": market_result["away_win_pct"],
                "overround": market_result["overround_pct"],
            }

        return {
            "home_win_pct": round(float(final_probs[2]) * 100, 1),
            "draw_pct": round(float(final_probs[1]) * 100, 1),
            "away_win_pct": round(float(final_probs[0]) * 100, 1),
            "most_likely_score": poisson_result["most_likely_score"],
            "expected_goals": (
                poisson_result["expected_home_goals"],
                poisson_result["expected_away_goals"],
            ),
            "confidence": confidence,
            "model_agreement": agreement,
            "model_spread_pct": round(model_spread * 100, 1),
            "individual_models": individual,
            "features_used": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in features.items()},
        }

    def sample_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
    ) -> Tuple[int, int]:
        """
        从预测分布采样一场比赛的比分（用于蒙特卡洛模拟）。

        返回
        -------
        Tuple[int, int]
            (主队进球, 客队进球)
        """
        return self.poisson.sample_match(home_team, away_team, is_neutral)

    def predict_print(self, home_team: str, away_team: str, **kwargs) -> Dict:
        """
        预测并格式化打印。
        """
        result = self.predict(home_team, away_team, **kwargs)

        print(f"\n{'='*55}")
        print(f"  {home_team} vs {away_team}")
        print(f"{'='*55}")
        print(f"  📊 最终预测:")
        print(f"    主胜: {result['home_win_pct']}%  |  平局: {result['draw_pct']}%  |  客胜: {result['away_win_pct']}%")
        print(f"  ⚽ 预期进球: {home_team} {result['expected_goals'][0]} - {result['expected_goals'][1]} {away_team}")
        print(f"  🎯 最常见比分: {result['most_likely_score']}")
        print(f"  🔒 置信度: {result['confidence']}  |  模型一致性: {result['model_agreement']}")
        print(f"{'='*55}")

        for name, pred in result["individual_models"].items():
            print(f"  {name:10s}: 主{pred['home_win']:5.1f}%  平{pred['draw']:5.1f}%  客{pred['away_win']:5.1f}%")

        return result


def load_predictor(
    poisson_path: str = None,
    elo_path: str = None,
    xgb_path: str = None,
    stacking_path: str = None,
    calibrator_path: str = None,
    team_attr_path: str = None,
    form_path: str = None,
) -> MatchPredictor:
    """
    从文件加载完整预测器。

    如果路径为 None，则创建空实例（后续手动设置）。
    """
    poisson = DixonColesModel()
    if poisson_path:
        with open(poisson_path, "rb") as f:
            poisson_data = pickle.load(f)
            poisson.mu = poisson_data["mu"]
            poisson.home_advantage = poisson_data["home_advantage"]
            poisson.rho = poisson_data["rho"]
            poisson.attack = poisson_data["attack"]
            poisson.defense = poisson_data["defense"]
            poisson.teams = poisson_data["teams"]
            poisson._fitted = True

    elo = EloRating()
    if elo_path:
        with open(elo_path, "rb") as f:
            elo_data = pickle.load(f)
            elo.ratings = elo_data["ratings"]
            elo.match_count = elo_data.get("match_count", {})

    xgb_model = XGBoostMatchModel()
    if xgb_path:
        xgb_model.load(xgb_path)

    stacking = StackingEnsemble()
    if stacking_path:
        stacking.load(stacking_path)

    calibrator = None
    if calibrator_path:
        calibrator = ProbabilityCalibrator()
        calibrator.load(calibrator_path)

    team_attributes = pd.read_csv(team_attr_path) if team_attr_path else None
    recent_form = pd.read_csv(form_path) if form_path else None

    return MatchPredictor(
        poisson_model=poisson,
        elo=elo,
        xgb_model=xgb_model,
        stacking=stacking,
        calibrator=calibrator,
        team_attributes=team_attributes,
        recent_form=recent_form,
    )


if __name__ == "__main__":
    print("单场预测引擎模块加载成功。")
    print("使用方法: predictor = MatchPredictor(poisson, elo, xgb, stacking)")
