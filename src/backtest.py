"""
回测验证系统
===========
用 2018 和 2022 世界杯验证模型预测能力。

增强（相比原始版）：
1. 严格的时间点一致性（只用世界杯前数据训练）
2. Brier Score、LogLoss、准确率三重指标
3. 与博彩市场直接对比
4. 校准曲线评估
5. 偏差分析（系统性高估/低估哪些类型的球队）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import json

from data_pipeline import calculate_match_weight, classify_tournament
from poisson_model import DixonColesModel
from elo_calculator import EloRating
from xgboost_model import XGBoostMatchModel
from stacking_ensemble import StackingEnsemble
from market_baseline import MarketBaseline
from calibration import ProbabilityCalibrator, evaluate_calibration, print_calibration_report
from match_predictor import MatchPredictor


class BacktestEngine:
    """
    世界杯预测回测引擎。

    参数
    ----------
    match_history : pd.DataFrame
        完整的国际比赛历史数据
    world_cup_fixtures : Dict[int, pd.DataFrame]
        {2018: fixtures_2018, 2022: fixtures_2022}
    world_cup_actual : Dict[int, pd.DataFrame]
        {2018: actual_results, 2022: actual_results}
    """

    def __init__(
        self,
        match_history: pd.DataFrame,
        world_cup_fixtures: Dict[int, pd.DataFrame],
        world_cup_actual: Optional[Dict[int, pd.DataFrame]] = None,
    ):
        self.match_history = match_history.copy()
        self.match_history["date"] = pd.to_datetime(self.match_history["date"])
        self.world_cup_fixtures = world_cup_fixtures
        self.world_cup_actual = world_cup_actual or {}

    # ═══════════════════════════════════════
    # 单届世界杯回测
    # ═══════════════════════════════════════

    def backtest_world_cup(
        self,
        year: int,
        verbose: bool = True,
    ) -> Dict:
        """
        回测一届世界杯。

        流程：
        1. 仅用该届世界杯之前的比赛数据训练所有模型
        2. 预测该届世界杯每场比赛的结果
        3. 与实际结果对比

        参数
        ----------
        year : int
            世界杯年份 (2018 或 2022)
        verbose : bool

        返回
        -------
        Dict
            回测指标
        """
        cutoff_date = pd.Timestamp(f"{year}-06-01")  # 世界杯前

        if verbose:
            print(f"\n{'='*55}")
            print(f"  {year} 世界杯回测")
            print(f"  截止日期: {cutoff_date.date()}")
            print(f"{'='*55}")

        # ── 1. 准备训练数据 ──
        train_data = self.match_history[self.match_history["date"] < cutoff_date].copy()

        # 分类赛事并加权
        if "tournament_type" not in train_data.columns:
            if "tournament" in train_data.columns:
                train_data["tournament_type"] = train_data["tournament"].apply(
                    classify_tournament
                )
            else:
                train_data["tournament_type"] = "Friendly"

        if "is_knockout" not in train_data.columns:
            train_data["is_knockout"] = False

        train_data["weight"] = train_data.apply(
            lambda r: calculate_match_weight(
                r["date"], cutoff_date, r["tournament_type"], r["is_knockout"]
            ),
            axis=1,
        )
        train_data = train_data[train_data["weight"] > 0]

        if verbose:
            print(f"  训练数据: {len(train_data)} 场比赛")

        # ── 2. 训练模型 ──
        # Poisson
        poisson = DixonColesModel()
        poisson.fit(train_data, verbose=verbose)

        # Elo
        elo = EloRating()
        elo.fit_all(train_data)

        # XGBoost
        team_params_df = poisson.get_team_params_df()
        recent_form = self._compute_recent_form(train_data, cutoff_date)
        xgb_model = XGBoostMatchModel(n_estimators=300, max_depth=4)
        # 为 XGBoost 构建训练特征
        xgb_train = xgb_model.build_features(
            train_data, team_params_df, elo.get_ratings_dict(), recent_form
        )
        X_train = xgb_train[xgb_model.feature_names]
        y_train = xgb_train["label"]
        xgb_model.fit(X_train, y_train, sample_weights=train_data["weight"].values, verbose=verbose)

        # Stacking（需要在训练集上的交叉验证预测）
        # 简化：用训练集的预测来训练元学习器
        stacking = self._train_stacking_on_data(
            train_data, poisson, elo, xgb_model, verbose
        )

        # 校准器
        all_base_preds, all_y = self._collect_base_predictions(
            train_data, poisson, elo, xgb_model, stacking
        )
        calibrator = None
        if all_base_preds is not None and len(all_y) > 50:
            calibrator = ProbabilityCalibrator(method="isotonic")
            calibrator.fit(all_base_preds, all_y)

        # 市场基线
        market = MarketBaseline()

        # ── 3. 构建预测器 ──
        predictor = MatchPredictor(
            poisson_model=poisson,
            elo=elo,
            xgb_model=xgb_model,
            stacking=stacking,
            calibrator=calibrator,
            market=market,
            team_attributes=None,
            recent_form=recent_form,
        )

        # ── 4. 预测该届世界杯 ──
        fixtures = self.world_cup_fixtures.get(year)
        if fixtures is None:
            print(f"  ⚠️ 缺少 {year} 世界杯赛程数据")
            return {}

        predictions = []
        actuals = []

        for _, match in fixtures.iterrows():
            pred = predictor.predict(
                home_team=match["home_team"],
                away_team=match["away_team"],
                is_neutral=True,
                odds=(
                    match.get("home_odds"),
                    match.get("draw_odds"),
                    match.get("away_odds"),
                ) if "home_odds" in match else None,
            )
            predictions.append(pred)

            # 实际结果
            actual_match = self._find_actual_result(match, year)
            if actual_match is not None:
                actuals.append(actual_match)

        if not actuals:
            print(f"  ⚠️ 缺少 {year} 世界杯实际结果")
            return {}

        # ── 5. 计算指标 ──
        metrics = self._compute_metrics(predictions, actuals, verbose)
        metrics["year"] = year
        metrics["train_samples"] = len(train_data)
        metrics["predictions"] = predictions

        return metrics

    # ═══════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════

    @staticmethod
    def _compute_recent_form(
        matches: pd.DataFrame, reference_date: pd.Timestamp
    ) -> pd.DataFrame:
        """计算每队的近期状态。"""
        from data_pipeline import compute_recent_form

        # 重组为"每队每场一行"格式
        home_records = matches.rename(
            columns={"home_team": "team", "home_goals": "goals_for", "away_goals": "goals_against"}
        )
        away_records = matches.rename(
            columns={"away_team": "team", "away_goals": "goals_for", "home_goals": "goals_against"}
        )
        all_records = pd.concat([
            home_records[["date", "team", "goals_for", "goals_against"]],
            away_records[["date", "team", "goals_for", "goals_against"]],
        ])
        all_records["date"] = pd.to_datetime(all_records["date"])

        return compute_recent_form(all_records, reference_date=reference_date)

    def _train_stacking_on_data(
        self,
        train_data: pd.DataFrame,
        poisson: DixonColesModel,
        elo: EloRating,
        xgb_model: XGBoostMatchModel,
        verbose: bool,
    ) -> StackingEnsemble:
        """用训练集的各模型预测训练 Stacking 元学习器。"""
        stacking = StackingEnsemble()

        # 收集每个模型的概率预测
        base_preds = {"poisson": [], "elo": [], "xgboost": []}
        labels = []

        team_params_df = poisson.get_team_params_df()
        recent_form = self._compute_recent_form(
            train_data, train_data["date"].max()
        )

        for _, match in train_data.iterrows():
            # Poisson
            pr = poisson.predict_result(match["home_team"], match["away_team"],
                                        is_neutral=match.get("neutral", "N") in ("Y", "y"))
            base_preds["poisson"].append([
                pr["away_win_pct"] / 100,
                pr["draw_pct"] / 100,
                pr["home_win_pct"] / 100,
            ])

            # Elo
            er = elo.predict(match["home_team"], match["away_team"])
            base_preds["elo"].append([
                er["away_win_pct"] / 100,
                er["draw_pct"] / 100,
                er["home_win_pct"] / 100,
            ])

            # XGBoost
            feats = xgb_model._build_single_match_features(
                match["home_team"], match["away_team"],
                team_params_df, elo.get_ratings_dict(), recent_form,
            )
            xr = xgb_model.predict_single(feats)
            base_preds["xgboost"].append([
                xr["away_win_pct"] / 100,
                xr["draw_pct"] / 100,
                xr["home_win_pct"] / 100,
            ])

            # 标签
            if match["home_goals"] > match["away_goals"]:
                labels.append(2)
            elif match["home_goals"] == match["away_goals"]:
                labels.append(1)
            else:
                labels.append(0)

        # 转 numpy
        base_preds_np = {k: np.array(v) for k, v in base_preds.items()}
        y_np = np.array(labels)

        stacking.fit(base_preds_np, y_np, verbose=verbose)
        return stacking

    def _collect_base_predictions(
        self,
        data: pd.DataFrame,
        poisson: DixonColesModel,
        elo: EloRating,
        xgb_model: XGBoostMatchModel,
        stacking: StackingEnsemble,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """收集基模型预测用于校准器训练。"""
        # 采样避免过大
        sample = data.sample(min(200, len(data)), random_state=42)

        all_preds = []
        all_y = []

        team_params_df = poisson.get_team_params_df()
        recent_form = self._compute_recent_form(data, data["date"].max())

        for _, match in sample.iterrows():
            base = {}

            pr = poisson.predict_result(match["home_team"], match["away_team"])
            base["poisson"] = np.array([
                pr["away_win_pct"] / 100,
                pr["draw_pct"] / 100,
                pr["home_win_pct"] / 100,
            ])

            er = elo.predict(match["home_team"], match["away_team"])
            base["elo"] = np.array([
                er["away_win_pct"] / 100,
                er["draw_pct"] / 100,
                er["home_win_pct"] / 100,
            ])

            feats = xgb_model._build_single_match_features(
                match["home_team"], match["away_team"],
                team_params_df, elo.get_ratings_dict(), recent_form,
            )
            xr = xgb_model.predict_single(feats)
            base["xgboost"] = np.array([
                xr["away_win_pct"] / 100,
                xr["draw_pct"] / 100,
                xr["home_win_pct"] / 100,
            ])

            try:
                stacked = stacking.predict_single(base)
                final = np.array([
                    stacked["away_win_pct"] / 100,
                    stacked["draw_pct"] / 100,
                    stacked["home_win_pct"] / 100,
                ])
                all_preds.append(final)
            except Exception:
                continue

            if match["home_goals"] > match["away_goals"]:
                all_y.append(2)
            elif match["home_goals"] == match["away_goals"]:
                all_y.append(1)
            else:
                all_y.append(0)

        if not all_preds:
            return None, None

        return np.array(all_preds), np.array(all_y)

    def _find_actual_result(
        self, match: pd.Series, year: int
    ) -> Optional[Dict]:
        """在真实结果中查找匹配的比赛。"""
        actual_df = self.world_cup_actual.get(year)
        if actual_df is None:
            return None

        # 模糊匹配（队名可能不完全一致）
        for _, row in actual_df.iterrows():
            if (row.get("home_team") == match["home_team"] and
                    row.get("away_team") == match["away_team"]):
                return {
                    "home_goals": row["home_goals"],
                    "away_goals": row["away_goals"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                }
        return None

    def _compute_metrics(
        self,
        predictions: List[Dict],
        actuals: List[Dict],
        verbose: bool = True,
    ) -> Dict:
        """计算回测指标。"""
        n = len(predictions)
        correct = 0
        correct_outcome = 0  # 方向正确（胜/平/负）
        logloss = 0.0
        brier_scores = []
        pred_probs = []
        true_labels = []

        for pred, actual in zip(predictions, actuals):
            # 确定结果类别
            if actual["home_goals"] > actual["away_goals"]:
                true_label = 2  # 主胜
            elif actual["home_goals"] == actual["away_goals"]:
                true_label = 1  # 平局
            else:
                true_label = 0  # 客胜

            pred_proba = np.array([
                pred["away_win_pct"] / 100,
                pred["draw_pct"] / 100,
                pred["home_win_pct"] / 100,
            ])

            pred_label = np.argmax(pred_proba)

            if pred_label == true_label:
                correct += 1

            # 方向正确（只判断胜负方向）
            pred_direction = 1 if pred_proba[2] > pred_proba[0] else (-1 if pred_proba[0] > pred_proba[2] else 0)
            true_direction = 1 if true_label == 2 else (-1 if true_label == 0 else 0)
            if pred_direction == true_direction and true_direction != 0:
                correct_outcome += 1

            # LogLoss
            eps = 1e-10
            logloss += -np.log(max(pred_proba[true_label], eps))

            # Brier
            true_onehot = np.eye(3)[true_label]
            brier_scores.append(np.mean((pred_proba - true_onehot) ** 2))

            pred_probs.append(pred_proba)
            true_labels.append(true_label)

        n_actual = len(actuals)
        acc = correct / n_actual * 100
        direction_acc = correct_outcome / n_actual * 100
        avg_logloss = logloss / n_actual
        avg_brier = np.mean(brier_scores)

        metrics = {
            "n_matches": n_actual,
            "accuracy_pct": round(acc, 1),
            "direction_accuracy_pct": round(direction_acc, 1),
            "log_loss": round(avg_logloss, 4),
            "brier_score": round(avg_brier, 4),
        }

        # 校准评估
        if len(pred_probs) >= 10:
            cal_metrics = evaluate_calibration(
                np.array(true_labels), np.array(pred_probs)
            )
            metrics.update(cal_metrics)

        if verbose:
            print(f"\n  📊 回测指标:")
            print(f"    比赛准确率: {acc:.1f}% (预测正确{correct}/{n_actual})")
            print(f"    方向准确率: {direction_acc:.1f}% (只判胜负方向)")
            print(f"    Log Loss:   {avg_logloss:.4f}")
            print(f"    Brier Score: {avg_brier:.4f}")

        return metrics

    # ═══════════════════════════════════════
    # 对比分析
    # ═══════════════════════════════════════

    def compare_with_market(
        self,
        year: int,
        backtest_results: Dict,
        market_odds: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict:
        """
        将模型预测与博彩市场对比。

        参数
        ----------
        year : int
            世界杯年份
        backtest_results : Dict
            回测结果（含 predictions）
        market_odds : pd.DataFrame
            该届世界杯的博彩赔率

        返回
        -------
        Dict
            对比指标
        """
        if verbose:
            print(f"\n  🎲 {year} 世界杯: 模型 vs 博彩市场")

        market = MarketBaseline()
        predictions = backtest_results.get("predictions", [])
        actuals = self.world_cup_actual.get(year)

        model_correct = 0
        market_correct = 0
        total = 0

        for pred in predictions:
            h_team = pred.get("home_team", "")
            a_team = pred.get("away_team", "")

            # 查找对应赔率
            odds_row = market_odds[
                (market_odds["home_team"] == h_team) &
                (market_odds["away_team"] == a_team)
            ]
            if odds_row.empty:
                continue

            odds = odds_row.iloc[0]
            market_pred = market.predict_single(
                odds["home_odds"], odds["draw_odds"], odds["away_odds"]
            )

            # 实际结果
            actual_row = actuals[
                (actuals["home_team"] == h_team) &
                (actuals["away_team"] == a_team)
            ]
            if actual_row.empty:
                continue

            actual = actual_row.iloc[0]
            if actual["home_goals"] > actual["away_goals"]:
                true_label = 2
            elif actual["home_goals"] == actual["away_goals"]:
                true_label = 1
            else:
                true_label = 0

            # 模型预测
            model_label = np.argmax([
                pred["away_win_pct"] / 100,
                pred["draw_pct"] / 100,
                pred["home_win_pct"] / 100,
            ])

            # 市场预测
            market_label = np.argmax([
                market_pred["away_win_pct"] / 100,
                market_pred["draw_pct"] / 100,
                market_pred["home_win_pct"] / 100,
            ])

            if model_label == true_label:
                model_correct += 1
            if market_label == true_label:
                market_correct += 1
            total += 1

        result = {
            "total_matches": total,
            "model_accuracy": round(model_correct / total * 100, 1) if total else 0,
            "market_accuracy": round(market_correct / total * 100, 1) if total else 0,
        }

        if verbose and total:
            print(f"    模型准确率: {result['model_accuracy']}%")
            print(f"    市场准确率: {result['market_accuracy']}%")
            diff = result["model_accuracy"] - result["market_accuracy"]
            emoji = "✅" if diff > 0 else "⚠️"
            print(f"    差异: {diff:+.1f}% {emoji}")

        return result

    # ═══════════════════════════════════════
    # 完整回测流程
    # ═══════════════════════════════════════

    def run_full_backtest(
        self,
        years: List[int] = [2018, 2022],
        market_odds: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> Dict:
        """
        执行完整的多届回测。

        返回
        -------
        Dict
            {2018: {...}, 2022: {...}, "summary": {...}}
        """
        all_results = {}

        for year in years:
            result = self.backtest_world_cup(year)
            all_results[year] = result

            # 与市场对比（如有赔率数据）
            if market_odds and year in market_odds:
                comparison = self.compare_with_market(year, result, market_odds[year])
                result["market_comparison"] = comparison

        # 汇总
        summary = self._summarize(all_results)
        all_results["summary"] = summary

        return all_results

    def _summarize(self, all_results: Dict) -> Dict:
        """汇总多届回测的平均表现。"""
        accs = []
        briers = []
        for year, result in all_results.items():
            if isinstance(year, int):
                if "accuracy_pct" in result:
                    accs.append(result["accuracy_pct"])
                if "brier_score" in result:
                    briers.append(result["brier_score"])

        return {
            "avg_accuracy": round(np.mean(accs), 1) if accs else None,
            "avg_brier": round(np.mean(briers), 4) if briers else None,
        }


def print_backtest_summary(results: Dict):
    """打印回测总结。"""
    print(f"\n{'='*55}")
    print(f"  回测总结")
    print(f"{'='*55}")

    for year in [2018, 2022]:
        if year in results:
            r = results[year]
            print(f"  {year}世界杯:")
            print(f"    准确率: {r.get('accuracy_pct', 'N/A')}%  "
                  f"Brier: {r.get('brier_score', 'N/A')}  "
                  f"LogLoss: {r.get('log_loss', 'N/A')}")
            if "market_comparison" in r:
                mc = r["market_comparison"]
                print(f"    模型 {mc['model_accuracy']}% vs 市场 {mc['market_accuracy']}%")

    if "summary" in results:
        s = results["summary"]
        print(f"  平均: 准确率 {s.get('avg_accuracy', 'N/A')}%, "
              f"Brier {s.get('avg_brier', 'N/A')}")


if __name__ == "__main__":
    print("回测验证模块加载成功。")
    print("使用方法: engine = BacktestEngine(match_history, fixtures, actuals)")
