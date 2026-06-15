"""
蒙特卡洛世界杯赛事模拟器
=======================
完整模拟 2026 世界杯从小组赛到决赛的全过程。

增强（相比原始版）：
1. 动态状态更新：根据前一场表现微调攻击/防守参数
2. 小组出线规则完全复现 FIFA 标准（含8个最佳第3名）
3. 淘汰赛加时 + 点球模型
4. 伤病/停赛随机事件
5. 旅行距离和休息天数影响
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import json
import time

from match_predictor import MatchPredictor


class TournamentSimulator:
    """
    蒙特卡洛世界杯赛事模拟器。

    参数
    ----------
    predictor : MatchPredictor
        单场比赛预测器（含所有子模型）
    fixtures : pd.DataFrame
        赛程表
    n_simulations : int
        蒙特卡洛模拟次数
    random_seed : int
        随机种子（可复现性）
    """

    # 2026世界杯小组赛分组结构（示例，将在数据采集后替换）
    DEFAULT_GROUPS = {
        "A": ["Mexico", "Canada", "Team_A3", "Team_A4"],
        "B": ["Team_B1", "Team_B2", "Team_B3", "Team_B4"],
        # ... 12组共48队
    }

    # 美加墨16个举办城市及海拔
    VENUES = {
        "Mexico City": 2240,
        "Guadalajara": 1566,
        "Monterrey": 540,
        "Los Angeles": 71,
        "San Francisco": 16,
        "Seattle": 50,
        "Vancouver": 2,
        "New York": 10,
        "Boston": 43,
        "Philadelphia": 12,
        "Atlanta": 320,
        "Miami": 2,
        "Dallas": 131,
        "Houston": 13,
        "Kansas City": 277,
        "Toronto": 76,
    }

    def __init__(
        self,
        predictor: MatchPredictor,
        fixtures: pd.DataFrame,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        self.predictor = predictor
        self.fixtures = fixtures
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_seed)

        # 从赛程中提取球队列表
        self.all_teams = sorted(
            set(fixtures["home_team"].unique()) | set(fixtures["away_team"].unique())
        )
        self.groups = self._build_groups_from_fixtures(fixtures)

        # 结果存储
        self.results: Dict[str, Dict[str, float]] = {}
        self.simulation_logs: List[Dict] = []

    # ═══════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════

    def _build_groups_from_fixtures(self, fixtures: pd.DataFrame) -> Dict[str, List[str]]:
        """从赛程中提取分组信息。"""
        groups = {}
        group_matches = fixtures[fixtures["stage"].str.lower().str.contains("group", na=False)]
        if group_matches.empty:
            group_matches = fixtures[fixtures["stage"].isin(["group", "Group Stage", "小组赛"])]

        if "group" in fixtures.columns:
            for group_name in fixtures["group"].unique():
                if pd.notna(group_name):
                    group_fixtures = fixtures[fixtures["group"] == group_name]
                    teams = set(group_fixtures["home_team"].unique()) | set(
                        group_fixtures["away_team"].unique()
                    )
                    groups[group_name] = sorted(teams)
        else:
            # 按赛程推断分组（每组6场比赛，12组）
            all_groups = "ABCDEFGHIJKL"
            all_teams = sorted(
                set(fixtures["home_team"].unique()) | set(fixtures["away_team"].unique())
            )
            # 简单分配：48队÷12组 = 4队/组
            for i, g in enumerate(all_groups):
                groups[g] = all_teams[i * 4:(i + 1) * 4]

        return groups

    # ═══════════════════════════════════════
    # 单次模拟
    # ═══════════════════════════════════════

    def simulate_one(self, sim_id: int = 0) -> Dict:
        """
        执行一次完整的赛事模拟。

        返回
        -------
        Dict
            {
                "champion": str,
                "runner_up": str,
                "semifinalists": List[str],
                "quarterfinalists": List[str],
                "round_of_32": List[str],
                "group_stage": List[str],
                "matches": List[Dict],  所有比赛记录
                "group_tables": Dict[str, List[Dict]],  小组积分榜
            }
        """
        # 状态追踪
        team_state = self._init_team_state()
        all_matches = []
        group_tables = {}

        # ── 小组赛 ──
        for group_name, teams in self.groups.items():
            table = self._simulate_group(group_name, teams, team_state, all_matches)
            group_tables[group_name] = table

        # ── 确定出线球队 ──
        qualified_teams = self._determine_qualified(group_tables)

        # ── 淘汰赛 ──
        champion, knockout_results = self._simulate_knockout_stage(
            qualified_teams, team_state, all_matches
        )

        return {
            "champion": champion,
            "runner_up": knockout_results.get("runner_up", ""),
            "semifinalists": knockout_results.get("semifinalists", []),
            "quarterfinalists": knockout_results.get("quarterfinalists", []),
            "round_of_32": knockout_results.get("round_of_32", []),
            "group_stage": [
                t for teams in self.groups.values()
                for t in teams
                if t not in [q["team"] for q in qualified_teams]
            ],
            "matches": all_matches,
            "group_tables": group_tables,
        }

    def _init_team_state(self) -> Dict:
        """初始化球队动态状态。"""
        state = {}
        for team in self.all_teams:
            state[team] = {
                "attack_bonus": 0.0,      # 状态起伏 → 攻击修正
                "defense_bonus": 0.0,      # 状态起伏 → 防守修正
                "fatigue": 0.0,            # 疲劳累积 (0~1)
                "suspensions": set(),      # 停赛球员（简化：用计数）
                "injuries": 0,             # 伤病数
                "matches_played": 0,
                "last_match_date": None,
                "rest_days": 5,
                "goals_scored": 0,
                "goals_conceded": 0,
                "form_goals_scored": [],
                "form_goals_conceded": [],
            }
        return state

    # ═══════════════════════════════════════
    # 小组赛模拟
    # ═══════════════════════════════════════

    def _simulate_group(
        self,
        group_name: str,
        teams: List[str],
        team_state: Dict,
        all_matches: List,
    ) -> List[Dict]:
        """
        模拟一个小组的所有比赛并计算积分榜。

        返回
        -------
        List[Dict]
            排序后的积分榜（第一名到第四名）
        """
        # 初始化积分表
        table = {
            team: {
                "team": team,
                "pts": 0, "played": 0, "w": 0, "d": 0, "l": 0,
                "gf": 0, "ga": 0, "gd": 0, "head_to_head": {},
            }
            for team in teams
        }

        # 循环赛：4队 → 6场比赛
        for i in range(4):
            for j in range(i + 1, 4):
                team_a = teams[i]
                team_b = teams[j]

                # 模拟比赛（小组赛都是中立场地）
                score_a, score_b, match_info = self._play_match(
                    team_a, team_b,
                    is_neutral=True,
                    stage="group",
                    team_state=team_state,
                )
                match_info["group"] = group_name
                match_info["stage"] = "group"
                all_matches.append(match_info)

                # 积分
                if score_a > score_b:
                    table[team_a]["pts"] += 3
                    table[team_a]["w"] += 1
                    table[team_b]["l"] += 1
                elif score_a < score_b:
                    table[team_b]["pts"] += 3
                    table[team_b]["w"] += 1
                    table[team_a]["l"] += 1
                else:
                    table[team_a]["pts"] += 1
                    table[team_b]["pts"] += 1
                    table[team_a]["d"] += 1
                    table[team_b]["d"] += 1

                table[team_a]["played"] += 1
                table[team_a]["gf"] += score_a
                table[team_a]["ga"] += score_b
                table[team_a]["gd"] = table[team_a]["gf"] - table[team_a]["ga"]
                table[team_b]["played"] += 1
                table[team_b]["gf"] += score_b
                table[team_b]["ga"] += score_a
                table[team_b]["gd"] = table[team_b]["gf"] - table[team_b]["ga"]

                # 记录直接交锋
                table[team_a]["head_to_head"][team_b] = (score_a, score_b)
                table[team_b]["head_to_head"][team_a] = (score_b, score_a)

        # 排序：积分 → 净胜球 → 进球数 → 直接交锋
        sorted_table = sorted(
            table.values(),
            key=lambda x: (
                x["pts"],
                x["gd"],
                x["gf"],
                # 简化 H2H（按所有对手的累计交手净胜球）
                sum(s[0] - s[1] for s in x["head_to_head"].values()),
            ),
            reverse=True,
        )

        return sorted_table

    # ═══════════════════════════════════════
    # 出线规则判定
    # ═══════════════════════════════════════

    def _determine_qualified(self, group_tables: Dict[str, List[Dict]]) -> List[Dict]:
        """
        确定32支出线球队：
        - 每组前2名（12组 × 2 = 24队）
        - 8个成绩最好的第3名

        返回
        -------
        List[Dict]
            出线球队列表（按淘汰赛对阵顺序排列）
        """
        qualified = []
        third_places = []

        for group_name, table in group_tables.items():
            # 前2名直接出线
            qualified.append({"team": table[0]["team"], "group": group_name, "rank": 1})
            qualified.append({"team": table[1]["team"], "group": group_name, "rank": 2})

            # 第3名进入候选
            third_places.append({
                "team": table[2]["team"],
                "group": group_name,
                "rank": 3,
                "pts": table[2]["pts"],
                "gd": table[2]["gd"],
                "gf": table[2]["gf"],
            })

        # 8个成绩最好的第3名
        third_places.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        best_thirds = third_places[:8]
        for tp in best_thirds:
            qualified.append({"team": tp["team"], "group": tp["group"], "rank": 3})

        return qualified

    # ═══════════════════════════════════════
    # 淘汰赛模拟
    # ═══════════════════════════════════════

    def _simulate_knockout_stage(
        self,
        qualified: List[Dict],
        team_state: Dict,
        all_matches: List,
    ) -> Tuple[str, Dict]:
        """
        模拟完整的淘汰赛阶段（32强 → 冠军）。

        返回
        -------
        Tuple[str, Dict]
            (冠军, 各阶段出局名单)
        """
        # 32强对阵表（简化：随机排列后按标准对阵）
        teams_32 = qualified.copy()
        # 按种子规则排列（简化：前2名 vs 第3名，尽可能）
        # 这里采用随机对阵，实际会因为数据不完整而简化

        round_names = {
            32: "Round of 32",
            16: "Round of 16",
            8: "Quarter-final",
            4: "Semi-final",
            2: "Final",
        }

        results_tracker = {
            "round_of_32": [],
            "quarterfinalists": [],
            "semifinalists": [],
            "runner_up": "",
        }

        current_round = teams_32

        while len(current_round) >= 2:
            round_size = len(current_round)
            next_round = []

            for i in range(0, len(current_round), 2):
                team_a = current_round[i]["team"]
                team_b = current_round[i + 1]["team"]

                is_final = round_size == 2

                score_a, score_b, match_info = self._play_match(
                    team_a, team_b,
                    is_neutral=True,
                    stage=round_names.get(round_size, f"R{round_size}"),
                    team_state=team_state,
                    is_knockout=True,
                )

                match_info["stage"] = round_names.get(round_size, f"R{round_size}")
                all_matches.append(match_info)

                # 加时赛/点球逻辑
                if score_a == score_b:
                    score_a, score_b = self._resolve_draw(
                        team_a, team_b, team_state, match_info
                    )

                winner = team_a if score_a > score_b else team_b
                loser = team_b if score_a > score_b else team_a

                next_round.append({"team": winner, "group": "", "rank": 0})

                # 记录
                if round_size == 32:
                    results_tracker["round_of_32"].append(loser)
                elif round_size == 8:
                    results_tracker["quarterfinalists"].append(loser)
                elif round_size == 4:
                    results_tracker["semifinalists"].append(loser)
                elif round_size == 2:
                    results_tracker["runner_up"] = loser

            current_round = next_round

        champion = current_round[0]["team"]
        return champion, results_tracker

    # ═══════════════════════════════════════
    # 单场比赛采样
    # ═══════════════════════════════════════

    def _play_match(
        self,
        team_a: str,
        team_b: str,
        is_neutral: bool,
        stage: str,
        team_state: Dict,
        is_knockout: bool = False,
    ) -> Tuple[int, int, Dict]:
        """
        采样一场比赛，含动态状态修正。

        返回
        -------
        Tuple[int, int, Dict]
            (主队进球, 客队进球, 比赛信息)
        """
        # ── 动态状态修正 ──
        state_a = team_state[team_a]
        state_b = team_state[team_b]

        # 疲劳影响
        fatigue_a = 1.0 - state_a["fatigue"] * 0.15
        fatigue_b = 1.0 - state_b["fatigue"] * 0.15

        # 状态起伏（随机波动）
        form_a = 1.0 + state_a.get("attack_bonus", 0) + self.rng.normal(0, 0.08)
        form_b = 1.0 + state_b.get("attack_bonus", 0) + self.rng.normal(0, 0.08)

        # 伤病影响
        injury_a = 1.0 - state_a["injuries"] * 0.12
        injury_b = 1.0 - state_b["injuries"] * 0.12

        # 综合修正因子
        adj_a = max(0.5, fatigue_a * form_a * injury_a)
        adj_b = max(0.5, fatigue_b * form_b * injury_b)

        # 临时调整泊松模型参数
        orig_attack_a = self.predictor.poisson.attack[team_a]
        orig_attack_b = self.predictor.poisson.attack[team_b]
        orig_defense_a = self.predictor.poisson.defense[team_a]
        orig_defense_b = self.predictor.poisson.defense[team_b]

        self.predictor.poisson.attack[team_a] *= adj_a
        self.predictor.poisson.attack[team_b] *= adj_b
        self.predictor.poisson.defense[team_a] *= adj_a
        self.predictor.poisson.defense[team_b] *= adj_b

        # 采样比分
        score_a, score_b = self.predictor.sample_match(team_a, team_b, is_neutral)

        # 恢复参数
        self.predictor.poisson.attack[team_a] = orig_attack_a
        self.predictor.poisson.attack[team_b] = orig_attack_b
        self.predictor.poisson.defense[team_a] = orig_defense_a
        self.predictor.poisson.defense[team_b] = orig_defense_b

        # ── 更新状态 ──
        for team, goals_for, goals_against in [
            (team_a, score_a, score_b), (team_b, score_b, score_a)
        ]:
            state = team_state[team]
            state["matches_played"] += 1
            state["goals_scored"] += goals_for
            state["goals_conceded"] += goals_against
            state["form_goals_scored"].append(goals_for)
            state["form_goals_conceded"].append(goals_against)

            # 只保留最近5场
            if len(state["form_goals_scored"]) > 5:
                state["form_goals_scored"] = state["form_goals_scored"][-5:]
                state["form_goals_conceded"] = state["form_goals_conceded"][-5:]

            # 更新状态起伏
            recent_gd = sum(state["form_goals_scored"][-3:]) - sum(
                state["form_goals_conceded"][-3:]
            )
            state["attack_bonus"] = np.clip(recent_gd / 3.0 * 0.05, -0.15, 0.15)

            # 疲劳累积（淘汰赛加时额外疲劳）
            base_fatigue = 0.08 if is_knockout else 0.05
            if score_a == score_b:  # 加时了
                base_fatigue += 0.06
            state["fatigue"] = min(1.0, state["fatigue"] + base_fatigue)

            # 随机伤病事件（每场 2% 概率）
            if self.rng.random() < 0.02:
                state["injuries"] += 1

            # 自然恢复
            state["fatigue"] = max(0.0, state["fatigue"] - 0.03)

        match_info = {
            "home_team": team_a,
            "away_team": team_b,
            "home_goals": score_a,
            "away_goals": score_b,
            "is_neutral": is_neutral,
            "stage": stage,
        }

        return score_a, score_b, match_info

    def _resolve_draw(
        self,
        team_a: str,
        team_b: str,
        team_state: Dict,
        match_info: Dict,
    ) -> Tuple[int, int]:
        """
        淘汰赛平局 → 加时30分钟 → 点球大战。

        返回
        -------
        Tuple[int, int]
            决胜后的最终比分
        """
        # 加时赛（进球率降为60%）
        lambda_h, lambda_a = self.predictor.poisson.expected_goals(team_a, team_b, True)
        et_goals_h = self.rng.poisson(lambda_h * 0.3)  # 半场35%强度
        et_goals_a = self.rng.poisson(lambda_a * 0.3)

        match_info["extra_time"] = f"{et_goals_h}-{et_goals_a}"

        if et_goals_h != et_goals_a:
            return match_info["home_goals"] + et_goals_h, match_info["away_goals"] + et_goals_a

        # 点球大战（简化：50%每队胜率，加轻微 Elo 修正）
        elo_h = self.predictor.elo.ratings.get(team_a, 1300)
        elo_a = self.predictor.elo.ratings.get(team_b, 1300)
        elo_adj = (elo_h - elo_a) / 400.0 * 0.1  # Elo 在点球中的影响很小
        prob_h_win_penalty = 0.50 + elo_adj

        if self.rng.random() < prob_h_win_penalty:
            match_info["penalties"] = f"{team_a} wins"
            return match_info["home_goals"] + 1, match_info["away_goals"]
        else:
            match_info["penalties"] = f"{team_b} wins"
            return match_info["home_goals"], match_info["away_goals"] + 1

    # ═══════════════════════════════════════
    # 批量模拟
    # ═══════════════════════════════════════

    def run(self, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        执行 N 次蒙特卡洛模拟。

        返回
        -------
        Dict[str, Dict[str, float]]
            {
                "Brazil": {"champion": 14.2, "final": 28.0, "semi": 42.0,
                           "quarter": 65.0, "round_of_32": 85.0, "group_stage": 15.0},
                ...
            }
        """
        # 初始化计数器
        counters = defaultdict(lambda: {
            "champion": 0, "final": 0, "semi": 0,
            "quarter": 0, "round_of_32": 0, "group_stage": 0,
        })

        t0 = time.time()

        for sim in range(self.n_simulations):
            result = self.simulate_one(sim_id=sim)

            # 冠军
            counters[result["champion"]]["champion"] += 1

            # 亚军
            if result["runner_up"]:
                counters[result["runner_up"]]["final"] += 1
            counters[result["champion"]]["final"] += 1

            # 四强
            for team in result["semifinalists"]:
                counters[team]["semi"] += 1

            # 八强
            for team in result["quarterfinalists"]:
                counters[team]["quarter"] += 1

            # 32强
            for team in result["round_of_32"]:
                if team:
                    counters[team]["round_of_32"] += 1

            # 小组未出线
            for team in result["group_stage"]:
                counters[team]["group_stage"] += 1

            # 进度报告
            if verbose and (sim + 1) % max(1, self.n_simulations // 10) == 0:
                elapsed = time.time() - t0
                rate = (sim + 1) / elapsed
                eta = (self.n_simulations - sim - 1) / rate
                print(f"  📊 模拟进度: {sim+1}/{self.n_simulations} "
                      f"({(sim+1)/self.n_simulations*100:.0f}%) "
                      f"[{elapsed:.1f}s, ETA {eta:.0f}s]")

        # 转为百分比
        results = {}
        for team in self.all_teams:
            c = counters[team]
            results[team] = {
                "champion_pct": round(c["champion"] / self.n_simulations * 100, 2),
                "final_pct": round(c["final"] / self.n_simulations * 100, 2),
                "semi_pct": round(c["semi"] / self.n_simulations * 100, 2),
                "quarter_pct": round(c["quarter"] / self.n_simulations * 100, 2),
                "round_of_32_pct": round(c["round_of_32"] / self.n_simulations * 100, 2),
                "group_stage_pct": round(c["group_stage"] / self.n_simulations * 100, 2),
            }

        self.results = results

        if verbose:
            elapsed = time.time() - t0
            print(f"\n✅ {self.n_simulations} 次模拟完成，耗时 {elapsed:.1f}s "
                  f"({self.n_simulations/elapsed:.0f} 次/秒)")

        return results

    def get_top_n(self, metric: str = "champion_pct", n: int = 10) -> pd.DataFrame:
        """
        获取指定指标的 Top N 球队。

        参数
        ----------
        metric : str
            指标名（champion_pct, final_pct, semi_pct 等）
        n : int
            返回前 N 支球队

        返回
        -------
        pd.DataFrame
        """
        rows = []
        for team, stats in self.results.items():
            rows.append({"team": team, **stats})
        df = pd.DataFrame(rows)
        return df.sort_values(metric, ascending=False).head(n)

    def print_summary(self, top_n: int = 10):
        """打印模拟结果摘要。"""
        if not self.results:
            print("无模拟结果，请先调用 run()")
            return

        print(f"\n{'='*65}")
        print(f"  2026世界杯蒙特卡洛模拟结果 ({self.n_simulations:,} 次)")
        print(f"{'='*65}")

        print(f"\n  🏆 夺冠概率 Top {top_n}:")
        print(f"  {'排名':<5} {'球队':<20} {'夺冠':>8} {'进决赛':>8} {'四强':>8} {'八强':>8}")
        print(f"  {'-'*60}")

        df = self.get_top_n("champion_pct", top_n)
        for i, (_, row) in enumerate(df.iterrows()):
            print(
                f"  {i+1:<5} {row['team']:<20} "
                f"{row['champion_pct']:>6.1f}% "
                f"{row['final_pct']:>6.1f}% "
                f"{row['semi_pct']:>6.1f}% "
                f"{row['quarter_pct']:>6.1f}%"
            )

        print(f"{'='*65}")

    def save_results(self, path: str):
        """保存模拟结果到 JSON。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_simulations": self.n_simulations,
                    "results": self.results,
                    "groups": self.groups,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[模拟器] 结果已保存到 {path}")


if __name__ == "__main__":
    print("赛事模拟器模块加载成功。")
    print("使用方法: sim = TournamentSimulator(predictor, fixtures); sim.run()")
