"""
Dixon-Coles 双变量泊松回归模型
==============================
基于 Dixon & Coles (1997) "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market"。

核心改进（相比独立泊松）：
1. 引入低比分相关性修正项 ρ，解决 0-0/1-0/0-1/1-1 被独立泊松低估的问题
2. 极大似然估计（MLE）拟合攻击/防守参数，带 ∑attack=0, ∑defense=0 约束
3. 支持带样本权重的加权似然（时间衰减 + 赛事重要性）
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class DixonColesModel:
    """
    Dixon-Coles 双变量泊松回归模型。

    模型设定
    --------
    log(λ_home) = μ + attack_home + defense_away + home_advantage
    log(λ_away) = μ + attack_away + defense_home

    约束：∑attack = 0, ∑defense = 0（以最后一个队为参照）

    低比分修正项 τ(λ_h, λ_a, x, y)：
        当 max(x, y) ≤ 1 时，ρ × D(x, y, λ_h, λ_a)
        其中 D = (1 - λ_h * λ_a * ρ_multiplier) 控制相关性方向和强度

    Dixon-Coles 对 0-0, 1-0, 0-1, 1-1 这四个比分施加修正：
        P(x, y) = τ_{λ_h,λ_a}(x, y) × Poisson(x|λ_h) × Poisson(y|λ_a)

    Attributes
    ----------
    mu : float
        全局进球基数
    home_advantage : float
        主场优势
    rho : float
        低比分相关系数（通常为负，表示低比分比独立模型预测的更多）
    attack : Dict[str, float]
        各队攻击强度
    defense : Dict[str, float]
        各队防守强度
    """

    def __init__(self, max_goals: int = 10):
        """
        参数
        ----------
        max_goals : int
            单队最大进球数（用于概率矩阵和模拟）
        """
        self.max_goals = max_goals
        self.mu = 0.5
        self.home_advantage = 0.3
        self.rho = -0.05  # 通常为小额负值
        self.attack: Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.teams: List[str] = []
        self._fitted = False

    # ═══════════════════════════════════════
    # 核心方法
    # ═══════════════════════════════════════

    def fit(
        self,
        matches: pd.DataFrame,
        team_cols: Tuple[str, str] = ("home_team", "away_team"),
        score_cols: Tuple[str, str] = ("home_goals", "away_goals"),
        weight_col: Optional[str] = "weight",
        neutral_col: Optional[str] = "neutral",
        verbose: bool = True,
    ) -> "DixonColesModel":
        """
        用极大似然估计拟合模型参数。

        参数
        ----------
        matches : pd.DataFrame
            历史比赛数据
        team_cols : Tuple[str, str]
            主客队列名
        score_cols : Tuple[str, str]
            主客队进球列名
        weight_col : str or None
            样本权重列名
        neutral_col : str or None
            中立场地标记列名（Y/N 或 1/0）
        verbose : bool
            是否输出拟合进度

        返回
        -------
        self
        """
        home_col, away_col = team_cols
        goals_h_col, goals_a_col = score_cols

        # 收集所有球队
        self.teams = sorted(set(matches[home_col].unique()) | set(matches[away_col].unique()))
        n_teams = len(self.teams)
        team_to_idx = {team: i for i, team in enumerate(self.teams)}

        # 权重
        if weight_col and weight_col in matches.columns:
            weights = matches[weight_col].values
        else:
            weights = np.ones(len(matches))

        # 中立场地
        if neutral_col and neutral_col in matches.columns:
            neutral = matches[neutral_col].isin(["Y", "y", 1, True, "1"]).astype(float).values
        else:
            neutral = np.zeros(len(matches))

        # 提取进球数据
        goals_h = matches[goals_h_col].astype(int).values
        goals_a = matches[goals_a_col].astype(int).values

        # 球队索引
        h_idx = np.array([team_to_idx[t] for t in matches[home_col]])
        a_idx = np.array([team_to_idx[t] for t in matches[away_col]])

        # ── 参数向量 ──
        # [mu, home_adv, rho, attack[0..n-2], defense[0..n-2]]
        # 最后球队的 attack[n-1] = -sum(attack[0..n-2])
        # 最后球队的 defense[n-1] = -sum(defense[0..n-2])
        n_params = 3 + 2 * (n_teams - 1)
        init_params = np.zeros(n_params)
        init_params[0] = 0.5   # mu
        init_params[1] = 0.3   # home_advantage
        init_params[2] = -0.05  # rho

        if verbose:
            print(f"[DixonColes] 拟合 {len(matches)} 场比赛, {n_teams} 支球队, {n_params} 个参数")

        result = minimize(
            self._neg_log_likelihood,
            init_params,
            args=(h_idx, a_idx, goals_h, goals_a, neutral, weights, n_teams),
            method="L-BFGS-B",
            bounds=[
                (-2.0, 2.0),    # mu
                (0.0, 0.8),     # home_advantage
                (-0.3, 0.3),    # rho
            ] + [(-3.0, 3.0)] * (2 * (n_teams - 1)),
            options={"maxiter": 5000, "ftol": 1e-12},
        )

        if not result.success and verbose:
            print(f"  [WARN] 优化收敛警告: {result.message}")

        # 提取参数
        self.mu = result.x[0]
        self.home_advantage = result.x[1]
        self.rho = result.x[2]

        a_raw = result.x[3 : 3 + n_teams - 1]
        d_raw = result.x[3 + n_teams - 1 :]

        # 补齐最后球队
        a_full = np.append(a_raw, -a_raw.sum())
        d_full = np.append(d_raw, -d_raw.sum())

        self.attack = {team: a_full[i] for i, team in enumerate(self.teams)}
        self.defense = {team: d_full[i] for i, team in enumerate(self.teams)}
        self._fitted = True

        if verbose:
            print(f"  ✅ 拟合完成: μ={self.mu:.3f}, home_adv={self.home_advantage:.3f}, ρ={self.rho:.4f}")
            self._print_top_teams(5)

        return self

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        h_idx: np.ndarray,
        a_idx: np.ndarray,
        goals_h: np.ndarray,
        goals_a: np.ndarray,
        neutral: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
    ) -> float:
        """加权负对数似然函数。"""
        mu = params[0]
        home_adv = params[1]
        rho = params[2]

        a_raw = params[3 : 3 + n_teams - 1]
        d_raw = params[3 + n_teams - 1 :]

        attack = np.append(a_raw, -a_raw.sum())
        defense = np.append(d_raw, -d_raw.sum())

        # 计算期望进球
        log_lambda_h = mu + attack[h_idx] + defense[a_idx] + home_adv * (1 - neutral)
        log_lambda_a = mu + attack[a_idx] + defense[h_idx]

        lambda_h = np.exp(np.clip(log_lambda_h, -10, 5))
        lambda_a = np.exp(np.clip(log_lambda_a, -10, 5))

        # 独立泊松对数概率
        ll_h = poisson.logpmf(goals_h, lambda_h)
        ll_a = poisson.logpmf(goals_a, lambda_a)
        ll = ll_h + ll_a

        # Dixon-Coles 低比分修正
        low_score_mask = (goals_h <= 1) & (goals_a <= 1)

        if low_score_mask.any():
            x = goals_h[low_score_mask]
            y = goals_a[low_score_mask]
            lh = lambda_h[low_score_mask]
            la = lambda_a[low_score_mask]

            tau = self._dc_adjustment(lh, la, x, y, rho)
            ll[low_score_mask] += np.log(np.maximum(tau, 1e-10))

        # 加权
        return -np.sum(weights * ll)

    @staticmethod
    def _dc_adjustment(
        lambda_h: np.ndarray,
        lambda_a: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Dixon-Coles 低比分修正项 τ。

        τ(λ_h, λ_a, x, y) = 1 - λ_h * λ_a * ρ    当 x=0, y=0
        τ(λ_h, λ_a, x, y) = 1 + λ_a * ρ          当 x=1, y=0
        τ(λ_h, λ_a, x, y) = 1 + λ_h * ρ          当 x=0, y=1
        τ(λ_h, λ_a, x, y) = 1 - ρ                当 x=1, y=1
        """
        tau = np.ones_like(lambda_h)

        # x=0, y=0
        mask_00 = (x == 0) & (y == 0)
        tau[mask_00] = 1.0 - lambda_h[mask_00] * lambda_a[mask_00] * rho

        # x=1, y=0
        mask_10 = (x == 1) & (y == 0)
        tau[mask_10] = 1.0 + lambda_a[mask_10] * rho

        # x=0, y=1
        mask_01 = (x == 0) & (y == 1)
        tau[mask_01] = 1.0 + lambda_h[mask_01] * rho

        # x=1, y=1
        mask_11 = (x == 1) & (y == 1)
        tau[mask_11] = 1.0 - rho

        return np.maximum(tau, 0.01)  # 不放缩到零

    # ═══════════════════════════════════════
    # 预测方法
    # ═══════════════════════════════════════

    def expected_goals(
        self, home_team: str, away_team: str, is_neutral: bool = False
    ) -> Tuple[float, float]:
        """
        计算预期进球数。

        返回
        -------
        Tuple[float, float]
            (主队预期进球, 客队预期进球)
        """
        if not self._fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        log_lambda_h = (
            self.mu
            + self.attack[home_team]
            + self.defense[away_team]
            + self.home_advantage * (1 - int(is_neutral))
        )
        log_lambda_a = self.mu + self.attack[away_team] + self.defense[home_team]

        return np.exp(log_lambda_h), np.exp(log_lambda_a)

    def score_probability_matrix(
        self, home_team: str, away_team: str, is_neutral: bool = False
    ) -> np.ndarray:
        """
        生成比分概率矩阵（含 Dixon-Coles 修正）。

        返回
        -------
        np.ndarray
            形状 (max_goals+1, max_goals+1) 的概率矩阵
            matrix[i, j] = P(主队进 i 球, 客队进 j 球)
        """
        lambda_h, lambda_a = self.expected_goals(home_team, away_team, is_neutral)

        # 独立泊松概率
        prob_h = poisson.pmf(np.arange(self.max_goals + 1), lambda_h)
        prob_a = poisson.pmf(np.arange(self.max_goals + 1), lambda_a)
        matrix = np.outer(prob_h, prob_a)

        # Dixon-Coles 修正（仅对 0-0, 1-0, 0-1, 1-1）
        tau_00 = 1.0 - lambda_h * lambda_a * self.rho
        tau_10 = 1.0 + lambda_a * self.rho
        tau_01 = 1.0 + lambda_h * self.rho
        tau_11 = 1.0 - self.rho

        matrix[0, 0] *= max(tau_00, 0.01)
        matrix[1, 0] *= max(tau_10, 0.01)
        matrix[0, 1] *= max(tau_01, 0.01)
        matrix[1, 1] *= max(tau_11, 0.01)

        # 重新归一化
        matrix /= matrix.sum()

        return matrix

    def predict_result(
        self, home_team: str, away_team: str, is_neutral: bool = False
    ) -> Dict:
        """
        预测比赛结果概率。

        返回
        -------
        Dict
            {
                "home_win_pct": float,  主胜概率 %
                "draw_pct": float,      平局概率 %
                "away_win_pct": float,  客胜概率 %
                "expected_home_goals": float,
                "expected_away_goals": float,
                "most_likely_score": str,
                "score_probability": float,  最常见比分概率 %
            }
        """
        matrix = self.score_probability_matrix(home_team, away_team, is_neutral)
        n = matrix.shape[0]

        # 胜平负概率
        # matrix[i,j] = P(主队进i球, 客队进j球)
        # 下三角 (i>j): 主胜  |  对角线 (i==j): 平局  |  上三角 (j>i): 客胜
        home_win = np.sum(np.tril(matrix, k=-1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, k=1))

        # 最常见比分
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

        lambda_h, lambda_a = self.expected_goals(home_team, away_team, is_neutral)

        # 归一化保证和为100%
        total = home_win + draw + away_win
        return {
            "home_win_pct": round(home_win / total * 100, 1),
            "draw_pct": round(draw / total * 100, 1),
            "away_win_pct": round(away_win / total * 100, 1),
            "expected_home_goals": round(lambda_h, 2),
            "expected_away_goals": round(lambda_a, 2),
            "most_likely_score": f"{max_idx[0]}-{max_idx[1]}",
            "score_probability": round(float(matrix[max_idx]) * 100, 1),
        }

    def sample_match(
        self, home_team: str, away_team: str, is_neutral: bool = False
    ) -> Tuple[int, int]:
        """
        从比分概率分布中随机采样一场比赛的比分。
        """
        matrix = self.score_probability_matrix(home_team, away_team, is_neutral)
        flat = matrix.flatten()
        idx = np.random.choice(len(flat), p=flat / flat.sum())
        n = matrix.shape[0]
        return idx // n, idx % n

    # ═══════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════

    def get_team_params_df(self) -> pd.DataFrame:
        """以 DataFrame 格式返回球队攻防参数。"""
        rows = []
        for team in self.teams:
            rows.append(
                {
                    "team": team,
                    "attack_strength": round(self.attack[team], 4),
                    "defense_strength": round(self.defense[team], 4),
                    "overall_rating": round(self.attack[team] - self.defense[team], 4),
                }
            )
        df = pd.DataFrame(rows)
        df = df.sort_values("overall_rating", ascending=False).reset_index(drop=True)
        return df

    def _print_top_teams(self, n: int = 5):
        """打印攻防最佳的球队。"""
        df = self.get_team_params_df()
        print("  📊 综合实力 Top {}:".format(n))
        for _, row in df.head(n).iterrows():
            print(
                f"    {row['team']:20s}  atk={row['attack_strength']:+.3f}  "
                f"def={row['defense_strength']:+.3f}  overall={row['overall_rating']:+.3f}"
            )


# ═══════════════════════════════════════════
# 简单封装：从 CSV 文件直接拟合
# ═══════════════════════════════════════════

def fit_model_from_csv(
    csv_path: str,
    reference_date: Optional[str] = None,
    verbose: bool = True,
) -> DixonColesModel:
    """
    从 CSV 文件读取历史比赛数据并拟合 Dixon-Coles 模型。

    参数
    ----------
    csv_path : str
        历史比赛 CSV（需含 home_team, away_team, home_goals, away_goals, date 列）
    reference_date : str or None
        回测基准日 "YYYY-MM-DD"，None 表示用最新日期
    verbose : bool

    返回
    -------
    DixonColesModel
        已拟合的模型
    """
    from data_pipeline import calculate_match_weight, classify_tournament

    matches = pd.read_csv(csv_path)
    matches["date"] = pd.to_datetime(matches["date"])

    if reference_date:
        ref_date = pd.to_datetime(reference_date)
    else:
        ref_date = matches["date"].max()

    # 分类赛事并计算权重
    if "tournament_type" not in matches.columns:
        if "tournament" in matches.columns:
            matches["tournament_type"] = matches["tournament"].apply(classify_tournament)
        else:
            matches["tournament_type"] = "Friendly"

    if "is_knockout" not in matches.columns:
        matches["is_knockout"] = False

    matches["weight"] = matches.apply(
        lambda r: calculate_match_weight(r["date"], ref_date, r["tournament_type"], r["is_knockout"]),
        axis=1,
    )

    # 过滤未来比赛
    matches = matches[matches["date"] <= ref_date].copy()

    model = DixonColesModel()
    model.fit(matches, verbose=verbose)

    return model


if __name__ == "__main__":
    print("DixonColes 模型模块加载成功。")
    print("使用方法: model = DixonColesModel(); model.fit(matches)")
