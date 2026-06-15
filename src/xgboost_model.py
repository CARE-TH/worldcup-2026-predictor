"""
XGBoost 分类模型
===============
用梯度提升树融合多维度特征，预测比赛胜平负。

增强（相比基础版本）：
1. 时间序列交叉验证（避免用未来信息预测过去）
2. 特征重要性分析
3. 类别权重处理（平局样本天生较少）
4. 概率输出校准友好
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict, List, Tuple, Optional
import json
import pickle
import warnings

warnings.filterwarnings("ignore")


class XGBoostMatchModel:
    """
    XGBoost 比赛预测模型。

    参数
    ----------
    objective : str
        目标函数
    n_estimators : int
        树的数量
    max_depth : int
        最大深度
    learning_rate : float
        学习率
    early_stopping_rounds : int
        早停轮数
    """

    def __init__(
        self,
        objective: str = "multi:softprob",
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 30,
    ):
        self.params = {
            "objective": objective,
            "num_class": 3,  # 0=客胜, 1=平局, 2=主胜
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "eval_metric": "mlogloss",
            "seed": 42,
            "verbosity": 0,
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self._fitted = False

    # ═══════════════════════════════════════
    # 特征构建
    # ═══════════════════════════════════════

    def build_features(
        self,
        matches: pd.DataFrame,
        team_params: pd.DataFrame,
        elo_ratings: Dict[str, float],
        recent_form: pd.DataFrame,
        team_attributes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        从历史比赛构建特征矩阵 X 和标签 y。

        参数
        ----------
        matches : pd.DataFrame
            历史比赛（含 home_team, away_team, home_goals, away_goals 等）
        team_params : pd.DataFrame
            攻防参数（来自泊松模型）
        elo_ratings : Dict[str, float]
            Elo 评分
        recent_form : pd.DataFrame
            近期状态
        team_attributes : pd.DataFrame or None
            球队属性

        返回
        -------
        pd.DataFrame
            特征矩阵（含 label 列）
        """
        rows = []

        for _, match in matches.iterrows():
            h = match["home_team"]
            a = match["away_team"]

            feats = self._build_single_match_features(
                h, a, team_params, elo_ratings, recent_form, team_attributes
            )

            # 标签
            if match["home_goals"] > match["away_goals"]:
                feats["label"] = 2  # 主胜
            elif match["home_goals"] == match["away_goals"]:
                feats["label"] = 1  # 平局
            else:
                feats["label"] = 0  # 客胜

            rows.append(feats)

        df = pd.DataFrame(rows)

        # 记录特征名（不含 label）
        self.feature_names = [c for c in df.columns if c != "label"]

        return df

    @staticmethod
    def _build_single_match_features(
        home_team: str,
        away_team: str,
        team_params: pd.DataFrame,
        elo_ratings: Dict[str, float],
        recent_form: pd.DataFrame,
        team_attributes: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """构建单场比赛特征（用于训练和预测）。"""
        f = {}

        # 攻击/防守参数差
        tp = team_params.set_index("team")
        if home_team in tp.index and away_team in tp.index:
            f["attack_diff"] = tp.loc[home_team, "attack_strength"] - tp.loc[away_team, "attack_strength"]
            f["defense_diff"] = tp.loc[home_team, "defense_strength"] - tp.loc[away_team, "defense_strength"]
            f["rating_diff"] = tp.loc[home_team, "overall_rating"] - tp.loc[away_team, "overall_rating"]
        else:
            f["attack_diff"] = 0.0
            f["defense_diff"] = 0.0
            f["rating_diff"] = 0.0

        # Elo 差
        elo_h = elo_ratings.get(home_team, 1300)
        elo_a = elo_ratings.get(away_team, 1300)
        f["elo_diff"] = elo_h - elo_a

        # 近期状态差
        for w in [5, 10]:
            h_form = recent_form[recent_form["team"] == home_team]
            a_form = recent_form[recent_form["team"] == away_team]
            col = f"form_{w}_win_rate"
            val_h = float(h_form[col].values[0]) if len(h_form) and col in h_form.columns and not pd.isna(h_form[col].values[0]) else 0.5
            val_a = float(a_form[col].values[0]) if len(a_form) and col in a_form.columns and not pd.isna(a_form[col].values[0]) else 0.5
            f[f"form_{w}_win_diff"] = val_h - val_a

        # 球队属性差
        if team_attributes is not None:
            ta = team_attributes.set_index("team")
            for col in ["fifa_rank", "squad_value_billion", "world_cup_appearances"]:
                if col in ta.columns and home_team in ta.index and away_team in ta.index:
                    if col == "fifa_rank":
                        f["rank_diff"] = ta.loc[away_team, col] - ta.loc[home_team, col]
                    elif col == "squad_value_billion":
                        f["value_diff_log"] = np.log1p(ta.loc[home_team, col]) - np.log1p(ta.loc[away_team, col])
                    else:
                        f[f"{col}_diff"] = ta.loc[home_team, col] - ta.loc[away_team, col]

        return f

    # ═══════════════════════════════════════
    # 训练
    # ═══════════════════════════════════════

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray] = None,
        n_splits: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        用时间序列交叉验证训练 XGBoost 模型。

        参数
        ----------
        X : pd.DataFrame
            特征矩阵（不含 label）
        y : pd.Series
            标签
        sample_weights : np.ndarray or None
            样本权重
        n_splits : int
            交叉验证折数
        verbose : bool

        返回
        -------
        Dict
            CV 评估指标
        """
        if verbose:
            print(f"[XGBoost] 训练 {len(X)} 个样本, {X.shape[1]} 个特征")

        # 类别权重
        class_weights = self._compute_class_weights(y)

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        best_model = None
        best_score = float("inf")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 样本权重
            sw = sample_weights[train_idx] if sample_weights is not None else None

            model = xgb.XGBClassifier(
                **self.params,
                n_estimators=self.n_estimators,
                scale_pos_weight=class_weights,
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=sw,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred_proba = model.predict_proba(X_val)
            score = log_loss(y_val, y_pred_proba)
            cv_scores.append(score)

            if score < best_score:
                best_score = score
                best_model = model

            if verbose:
                acc = accuracy_score(y_val, np.argmax(y_pred_proba, axis=1))
                print(f"  Fold {fold+1}/{n_splits}: LogLoss={score:.4f}, Acc={acc:.3f}")

        self.model = best_model
        self._fitted = True

        # 特征重要性
        self.feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        cv_mean = np.mean(cv_scores)
        if verbose:
            print(f"  ✅ XGBoost 训练完成: CV LogLoss={cv_mean:.4f} ± {np.std(cv_scores):.4f}")

        return {"cv_logloss_mean": cv_mean, "cv_logloss_std": np.std(cv_scores)}

    # ═══════════════════════════════════════
    # 预测
    # ═══════════════════════════════════════

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率。

        返回
        -------
        np.ndarray
            形状 (n_samples, 3), 列顺序: [客胜, 平局, 主胜]
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        return self.model.predict_proba(X[self.feature_names])

    def predict_single(self, features: Dict) -> Dict[str, float]:
        """
        预测单场比赛。

        参数
        ----------
        features : Dict
            特征字典

        返回
        -------
        Dict
            {"away_win_pct", "draw_pct", "home_win_pct"}
        """
        X = pd.DataFrame([features])[self.feature_names]
        proba = self.predict_proba(X)[0]
        return {
            "away_win_pct": round(proba[0] * 100, 1),
            "draw_pct": round(proba[1] * 100, 1),
            "home_win_pct": round(proba[2] * 100, 1),
        }

    # ═══════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════

    def save(self, path: str):
        """保存模型到文件。"""
        if self.model is None:
            raise RuntimeError("无模型可保存")
        # 保存完整对象（含 feature_names）
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "feature_importance": self.feature_importance,
                    "params": self.params,
                },
                f,
            )
        print(f"[XGBoost] 模型已保存到 {path}")

    def load(self, path: str):
        """从文件加载模型。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importance = data.get("feature_importance")
        self.params = data.get("params", self.params)
        self._fitted = True
        print(f"[XGBoost] 模型已加载 ({len(self.feature_names)} 个特征)")

    # ═══════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════

    @staticmethod
    def _compute_class_weights(y: pd.Series) -> Dict[int, float]:
        """计算类别权重以处理不平衡。"""
        counts = y.value_counts()
        total = len(y)
        weights = {}
        for cls, cnt in counts.items():
            weights[int(cls)] = total / (3 * cnt)
        return weights


if __name__ == "__main__":
    print("XGBoost 模型模块加载成功。")
    print("使用方法: model = XGBoostMatchModel(); model.fit(X, y)")
