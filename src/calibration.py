"""
概率校准模块
===========
对模型输出的原始概率进行校准，使概率估计更可靠。

方法
----
1. Platt Scaling（单参数 sigmoid 校准）——适合二分类
2. Isotonic Regression（非参数）——更灵活但需更多数据
3. Temperature Scaling —— 简单的温度参数平滑（多分类友好）

对于三分类问题（胜/平/负），使用一对多（One-vs-Rest）策略
或 Temperature Scaling 进行整体校准。

评估
----
1. Brier Score —— 概率预测的整体准确性
2. 校准曲线 —— 概率 vs 实际频率
3. ECE (Expected Calibration Error) —— 校准误差
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Dict, Tuple, Optional
import pickle
import warnings

warnings.filterwarnings("ignore")


class ProbabilityCalibrator:
    """
    概率校准器。

    参数
    ----------
    method : str
        校准方法: "temperature" / "isotonic" / "platt"
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrators = []  # 每类一个校准器
        self.temperature: float = 1.0  # Temperature Scaling 参数
        self._fitted = False

    def fit(
        self,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        拟合校准器。

        参数
        ----------
        y_pred_proba : np.ndarray
            原始模型预测概率 (n_samples, n_classes)
        y_true : np.ndarray
            真实标签 (n_samples,)，整数类别标签
        """
        n_classes = y_pred_proba.shape[1]

        if self.method == "temperature":
            self._fit_temperature(y_pred_proba, y_true)
        elif self.method == "isotonic":
            self._fit_isotonic(y_pred_proba, y_true, n_classes)
        elif self.method == "platt":
            self._fit_platt(y_pred_proba, y_true, n_classes)
        else:
            raise ValueError(f"未知校准方法: {self.method}")

        self._fitted = True
        return self

    def _fit_temperature(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
        """
        Temperature Scaling。

        最小化 NLL 找到最优温度 T:
            q_i = softmax(log(p_i) / T)
        """
        from scipy.optimize import minimize

        y_true_onehot = np.eye(y_pred_proba.shape[1])[y_true]

        def nll(T):
            T = T[0]
            if T <= 0.01:
                return 1e10
            logits = np.log(np.clip(y_pred_proba, 1e-10, 1.0)) / T
            # softmax
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            return -np.sum(y_true_onehot * np.log(np.clip(probs, 1e-10, 1.0)))

        result = minimize(nll, x0=[1.0], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        self.temperature = result.x[0]
        print(f"[校准] Temperature Scaling: T={self.temperature:.3f}")

    def _fit_isotonic(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray, n_classes: int
    ):
        """Isotonic Regression 校准（每类独立）。"""
        self.calibrators = []
        y_true_onehot = np.eye(n_classes)[y_true]

        for c in range(n_classes):
            iso = IsotonicRegression(
                y_min=0.001, y_max=0.999, out_of_bounds="clip"
            )
            iso.fit(y_pred_proba[:, c], y_true_onehot[:, c])
            self.calibrators.append(iso)

        print(f"[校准] Isotonic Regression: {n_classes} 个校准器已拟合")

    def _fit_platt(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray, n_classes: int
    ):
        """Platt Scaling（用 Logistic Regression 校准，一对多）。"""
        from sklearn.linear_model import LogisticRegression

        self.calibrators = []
        y_true_onehot = np.eye(n_classes)[y_true]

        for c in range(n_classes):
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(y_pred_proba[:, c].reshape(-1, 1), y_true_onehot[:, c])
            self.calibrators.append(lr)

        print(f"[校准] Platt Scaling: {n_classes} 个校准器已拟合")

    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        校准概率。

        参数
        ----------
        y_pred_proba : np.ndarray
            原始模型预测概率 (n_samples, n_classes)

        返回
        -------
        np.ndarray
            校准后的概率 (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("校准器尚未拟合")

        if self.method == "temperature":
            logits = np.log(np.clip(y_pred_proba, 1e-10, 1.0)) / self.temperature
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        elif self.method == "isotonic":
            calibrated = np.column_stack(
                [cal.predict(y_pred_proba[:, c]) for c, cal in enumerate(self.calibrators)]
            )
            # 归一化
            calibrated /= calibrated.sum(axis=1, keepdims=True)
        elif self.method == "platt":
            calibrated = np.column_stack(
                [cal.predict_proba(y_pred_proba[:, c].reshape(-1, 1))[:, 1]
                 for c, cal in enumerate(self.calibrators)]
            )
            calibrated /= calibrated.sum(axis=1, keepdims=True)
        else:
            calibrated = y_pred_proba

        return calibrated

    def save(self, path: str):
        """保存校准器。"""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "calibrators": self.calibrators,
                    "temperature": self.temperature,
                },
                f,
            )
        print(f"[校准] 校准器已保存到 {path}")

    def load(self, path: str):
        """加载校准器。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.method = data["method"]
        self.calibrators = data["calibrators"]
        self.temperature = data["temperature"]
        self._fitted = True
        print(f"[校准] 校准器已加载 ({self.method})")


# ═══════════════════════════════════════
# 校准评估工具
# ═══════════════════════════════════════

def evaluate_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    评估概率校准质量。

    参数
    ----------
    y_true : np.ndarray
        真实标签
    y_pred_proba : np.ndarray
        预测概率 (n_samples, n_classes)
    n_bins : int
        校准曲线分箱数

    返回
    -------
    Dict
        {
            "brier_score": float,
            "ece": float,  Expected Calibration Error
            "per_class_brier": List[float],
            "calibration_curve": (fraction_positives, mean_predicted),
        }
    """
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]

    # Brier Score（多类）
    brier = brier_score_loss(y_true_onehot.flatten(), y_pred_proba.flatten())

    # Per-class Brier
    per_class_brier = []
    for c in range(n_classes):
        per_class_brier.append(
            brier_score_loss(y_true_onehot[:, c], y_pred_proba[:, c])
        )

    # ECE (Expected Calibration Error)
    ece = _compute_ece(y_true, y_pred_proba, n_bins)

    # 校准曲线（只对主胜类别）
    frac_pos, mean_pred = calibration_curve(
        y_true_onehot[:, 2], y_pred_proba[:, 2], n_bins=n_bins
    )

    return {
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "per_class_brier": [round(b, 4) for b in per_class_brier],
        "calibration_curve": (frac_pos, mean_pred),
    }


def _compute_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error。

    将预测概率分箱，计算每箱内预测均值与实际频率之差的加权平均。
    """
    n_samples = y_pred_proba.shape[0]
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_conf = confidences[in_bin].mean()
            bin_acc = accuracies[in_bin].mean()
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return ece


def print_calibration_report(metrics: Dict):
    """打印校准评估报告。"""
    print("\n" + "=" * 55)
    print("  概率校准评估报告")
    print("=" * 55)
    print(f"  Brier Score : {metrics['brier_score']:.4f}  (越低越好, 0=完美)")
    print(f"  ECE         : {metrics['ece']:.4f}  (Expected Calibration Error)")
    print(f"  各类 Brier  : 客胜={metrics['per_class_brier'][0]:.4f}  "
          f"平局={metrics['per_class_brier'][1]:.4f}  主胜={metrics['per_class_brier'][2]:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    print("概率校准模块加载成功。")
    print("使用方法: cal = ProbabilityCalibrator(); cal.fit(y_pred, y_true)")
