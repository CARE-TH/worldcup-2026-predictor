"""
Stacking 集成模型
================
用元学习器（Logistic Regression）融合泊松/Elo/XGBoost/博彩市场
四个基模型的输出概率，自动学习最优加权。

替代原方案中主观的 0.35/0.20/0.45 固定权重。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict, List, Tuple, Optional
import pickle
import warnings

warnings.filterwarnings("ignore")


class StackingEnsemble:
    """
    Stacking 集成：用元学习器融合多个基模型。

    架构
    ----
    基模型 → 各输出 3 个概率（客胜/平局/主胜）
    元特征 → 拼接所有基模型预测（3 × N_models 维）
    元学习器 → LogisticRegression 输出最终 3 类概率

    Attributes
    ----------
    meta_model : LogisticRegression
        元学习器
    model_names : List[str]
        基模型名称列表
    """

    def __init__(self):
        self.meta_model: Optional[LogisticRegression] = None
        self.model_names: List[str] = []
        self._fitted = False

    def fit(
        self,
        base_predictions: Dict[str, np.ndarray],
        y: np.ndarray,
        n_splits: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        训练元学习器。

        参数
        ----------
        base_predictions : Dict[str, np.ndarray]
            各基模型的概率预测，shape 均为 (n_samples, 3)
            key: 模型名, value: 概率矩阵
        y : np.ndarray
            真实标签 (0=客胜, 1=平局, 2=主胜)
        n_splits : int
            交叉验证折数
        verbose : bool

        返回
        -------
        Dict
            CV 评估指标
        """
        self.model_names = sorted(base_predictions.keys())
        n_samples = len(y)

        # 拼接所有基模型预测为元特征
        meta_X = np.hstack([base_predictions[name] for name in self.model_names])

        if verbose:
            print(f"[Stacking] 融合 {len(self.model_names)} 个基模型:")
            for name in self.model_names:
                print(f"  - {name}")
            print(f"  元特征维度: {meta_X.shape[1]} ({len(self.model_names)} 模型 × 3 类)")

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=min(n_splits, n_samples - 1))
        cv_scores = []
        best_model = None
        best_score = float("inf")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(meta_X)):
            X_train, X_val = meta_X[train_idx], meta_X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Logistic Regression 元学习器
            # multinomial + LBFGS 支持多分类概率输出
            meta = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=42,
            )
            meta.fit(X_train, y_train)

            y_pred_proba = meta.predict_proba(X_val)
            score = log_loss(y_val, y_pred_proba)
            cv_scores.append(score)

            if score < best_score:
                best_score = score
                best_model = meta

            if verbose:
                acc = accuracy_score(y_val, np.argmax(y_pred_proba, axis=1))
                print(f"  Fold {fold+1}: LogLoss={score:.4f}, Acc={acc:.3f}")

        self.meta_model = best_model
        self._fitted = True

        # 提取权重解释
        self._print_weights(verbose)

        return {
            "cv_logloss_mean": np.mean(cv_scores),
            "cv_logloss_std": np.std(cv_scores),
        }

    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        融合基模型预测。

        参数
        ----------
        base_predictions : Dict[str, np.ndarray]
            各基模型的概率预测

        返回
        -------
        np.ndarray
            融合后的概率 (n_samples, 3)
        """
        if not self._fitted:
            raise RuntimeError("Stacking 集成尚未训练")

        meta_X = np.hstack(
            [base_predictions[name] for name in self.model_names if name in base_predictions]
        )
        return self.meta_model.predict_proba(meta_X)

    def predict_single(
        self, base_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        融合单场比赛预测。

        参数
        ----------
        base_predictions : Dict[str, np.ndarray]
            {"poisson": array([0.25, 0.30, 0.45]), "elo": ..., ...}

        返回
        -------
        Dict
            {"away_win_pct", "draw_pct", "home_win_pct", "model_weights"}
        """
        # 重组为单样本
        preds = {k: v.reshape(1, -1) for k, v in base_predictions.items()}
        proba = self.predict_proba(preds)[0]

        return {
            "away_win_pct": round(proba[0] * 100, 1),
            "draw_pct": round(proba[1] * 100, 1),
            "home_win_pct": round(proba[2] * 100, 1),
        }

    def get_weights(self) -> pd.DataFrame:
        """
        提取元学习器权重（近似解释）。

        返回每个基模型对每种类别的平均贡献权重。
        """
        if not self._fitted:
            return pd.DataFrame()

        coef = self.meta_model.coef_  # shape: (3, 3*N_models)
        n_models = len(self.model_names)

        rows = []
        for i, name in enumerate(self.model_names):
            # 模型 i 的 3 个概率对应 3 个输出类别的系数
            w_away = coef[0, i * 3:(i + 1) * 3].mean()  # 对 "客胜" 的平均影响
            w_draw = coef[1, i * 3:(i + 1) * 3].mean()  # 对 "平局" 的平均影响
            w_home = coef[2, i * 3:(i + 1) * 3].mean()  # 对 "主胜" 的平均影响
            rows.append(
                {
                    "model": name,
                    "weight_away": round(w_away, 4),
                    "weight_draw": round(w_draw, 4),
                    "weight_home": round(w_home, 4),
                    "importance": round(abs(w_away) + abs(w_draw) + abs(w_home), 4),
                }
            )

        df = pd.DataFrame(rows)
        # 归一化重要性
        total = df["importance"].sum()
        if total > 0:
            df["importance_norm"] = df["importance"] / total
        return df.sort_values("importance", ascending=False)

    def _print_weights(self, verbose: bool):
        """打印基模型的重要性。"""
        if not verbose:
            return
        weights = self.get_weights()
        if len(weights):
            print("  📊 基模型重要性:")
            for _, row in weights.iterrows():
                pct = row.get("importance_norm", 0) * 100
                print(f"    {row['model']:15s}: {pct:.1f}%")

    def save(self, path: str):
        """保存元学习器。"""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "meta_model": self.meta_model,
                    "model_names": self.model_names,
                },
                f,
            )
        print(f"[Stacking] 元学习器已保存到 {path}")

    def load(self, path: str):
        """加载元学习器。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.meta_model = data["meta_model"]
        self.model_names = data["model_names"]
        self._fitted = True
        print(f"[Stacking] 元学习器已加载 ({len(self.model_names)} 个基模型)")


if __name__ == "__main__":
    print("Stacking 集成模块加载成功。")
