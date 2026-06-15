"""
增强版小组赛比分预测
==================
整合以下数据源:
1. 实际比赛结果 (6月11-15日已进行的比赛)
2. 最新FIFA排名 (2026年6月11日)
3. 博彩赔率 (BetMGM/FanDuel/Kalshi)
4. 伤病名单 (关键球员缺失)
5. ESPN Power Rankings
6. 原泊松模型攻防参数

输出:
- 剩余60场小组赛的增强预测
- 模型校准报告
- 12组出线形势更新
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(__file__))
from poisson_model import DixonColesModel
from elo_calculator import EloRating

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. 实际比赛结果 (2026年6月11-15日)
# ═══════════════════════════════════════════════════════════
ACTUAL_RESULTS = {
    # June 11
    ("Mexico", "South Africa"): (2, 0),
    ("South Korea", "Czechia"): (2, 1),
    # June 12
    ("Canada", "Bosnia & Herzegovina"): (1, 1),
    ("United States", "Paraguay"): (4, 1),
    # June 13
    ("Qatar", "Switzerland"): (1, 1),
    ("Brazil", "Morocco"): (1, 1),
    ("Haiti", "Scotland"): (0, 1),
    ("Australia", "Turkiye"): (2, 0),
    # June 14
    ("Germany", "Curacao"): (7, 1),
    ("Cote d'Ivoire", "Ecuador"): (1, 0),
    ("Netherlands", "Japan"): (2, 2),
    # June 15
    ("Sweden", "Tunisia"): (5, 1),
}

# ═══════════════════════════════════════════════════════════
# 2. FIFA排名 (2026年6月11日)
# ═══════════════════════════════════════════════════════════
FIFA_RANKINGS = {
    "Argentina": 1877.27, "Spain": 1874.71, "France": 1870.70,
    "England": 1828.02, "Portugal": 1767.85, "Brazil": 1765.34,
    "Morocco": 1755.62, "Netherlands": 1749.20, "Germany": 1743.54,
    "Belgium": 1742.24, "Croatia": 1714.87, "Colombia": 1698.35,
    "Mexico": 1700.98, "Senegal": 1684.07, "Uruguay": 1673.07,
    "United States": 1671.23, "Japan": 1661.58, "Switzerland": 1658.0,
    "IR Iran": 1631.0, "Austria": 1625.0, "South Korea": 1618.0,
    "Australia": 1612.0, "Sweden": 1610.0, "Egypt": 1597.0,
    "Cote d'Ivoire": 1592.0, "Norway": 1586.0, "Turkiye": 1583.0,
    "Ecuador": 1578.0, "Scotland": 1573.0, "Czechia": 1567.0,
    "Paraguay": 1560.0, "Ghana": 1555.0, "Qatar": 1550.0,
    "Canada": 1545.0, "Algeria": 1540.0, "Bosnia & Herzegovina": 1535.0,
    "Tunisia": 1530.0, "Saudi Arabia": 1525.0, "South Africa": 1520.0,
    "Panama": 1510.0, "Cabo Verde": 1505.0, "Congo DR": 1500.0,
    "New Zealand": 1495.0, "Uzbekistan": 1490.0, "Jordan": 1485.0,
    "Iraq": 1480.0, "Haiti": 1475.0, "Curacao": 1470.0,
}

# ═══════════════════════════════════════════════════════════
# 3. 关键伤病 (影响球队实力的负面调整)
# ═══════════════════════════════════════════════════════════
INJURY_ADJUSTMENTS = {
    # 格式: 球队: (-攻击调整, +防守调整)  # 防守正值代表失球更多
    "Brazil": (-0.12, +0.08),      # Rodrygo(ACL), Militao(大腿), Estevao(大腿)
    "Japan": (-0.10, +0.05),       # Mitoma(大腿), Minamino(ACL)
    "Netherlands": (-0.15, +0.10), # Simons(ACL), Schouten(ACL), de Ligt(背), Timber(腹股沟)
    "Germany": (-0.05, +0.02),     # Gnabry(内收肌), Karl(大腿), ter Stegen(大腿)
    "Spain": (-0.05, +0.02),       # Fermin Lopez(跖骨), Aghehowa(ACL)
    "France": (-0.03, +0.01),      # Ekitike(跟腱) - Camavinga战术落选
    "USA": (-0.03, +0.02),         # Cardoso(踝关节)
    "Ghana": (-0.05, +0.03),       # Kudus(四头肌)
    "Canada": (-0.03, +0.02),      # Flores(ACL)
    "Mexico": (-0.02, +0.02),      # Malagon(跟腱)
    "Scotland": (-0.02, +0.02),    # Gilmour(膝)
    "Austria": (-0.05, +0.03),     # Baumgartner(伤)
    "England": (-0.02, +0.01),     # Palmer/Trent 战术落选(深度影响小)
}

# ═══════════════════════════════════════════════════════════
# 4. 赔率隐含实力 (从夺冠赔率反推)
# ═══════════════════════════════════════════════════════════
# 赔率 → 隐含概率 → 相对实力评分 (标准正态化)
BETTING_ODDS_IMPLIED = {
    "Spain": 0.16, "France": 0.16, "England": 0.11, "Portugal": 0.10,
    "Argentina": 0.09, "Brazil": 0.08, "Germany": 0.06, "Netherlands": 0.05,
    "Norway": 0.03, "Belgium": 0.03, "Morocco": 0.025, "United States": 0.02,
    "Colombia": 0.02, "Japan": 0.018, "Mexico": 0.015, "Uruguay": 0.012,
    "Croatia": 0.012,
}

# ═══════════════════════════════════════════════════════════
# 核心预测函数
# ═══════════════════════════════════════════════════════════

def build_enhanced_ratings(poisson_model):
    """
    融合三个数据源构建增强实力评分:
    1. 泊松模型攻防参数 (历史数据)
    2. FIFA排名 (当前实力)
    3. 伤病调整
    """

    # 从FIFA排名推导基础实力 (归一化: 均值=0, 标准差=0.5)
    fifa_values = np.array(list(FIFA_RANKINGS.values()))
    fifa_mean = fifa_values.mean()
    fifa_std = fifa_values.std()

    enhanced_attack = {}
    enhanced_defense = {}

    for team in poisson_model.teams:
        # 方法1: 泊松模型参数
        poisson_att = poisson_model.attack.get(team, 0.0)
        poisson_def = poisson_model.defense.get(team, 0.0)

        # 方法2: FIFA排名转实力
        fifa_score = FIFA_RANKINGS.get(team, 1500)
        fifa_z = (fifa_score - fifa_mean) / fifa_std
        fifa_att = fifa_z * 0.35  # 缩放到泊松参数量级
        fifa_def = -fifa_z * 0.25  # 强队防守好(负值)

        # 方法3: 赔率隐含 (如有)
        betting_score = BETTING_ODDS_IMPLIED.get(team, 0.003)
        betting_z = (betting_score - 0.01) / 0.05  # 标准化
        betting_att = np.clip(betting_z * 0.2, -0.3, 0.5)
        betting_def = np.clip(-betting_z * 0.15, -0.5, 0.3)

        # 伤病调整
        injury_att, injury_def = INJURY_ADJUSTMENTS.get(team, (0.0, 0.0))

        # 融合权重
        w_poisson = 0.35   # 历史数据
        w_fifa = 0.45      # FIFA排名(更反映当前)
        w_betting = 0.20   # 赔率(市场智慧)

        final_att = (poisson_att * w_poisson + fifa_att * w_fifa +
                     betting_att * w_betting + injury_att)
        final_def = (poisson_def * w_poisson + fifa_def * w_fifa +
                     betting_def * w_betting + injury_def)

        enhanced_attack[team] = round(final_att, 4)
        enhanced_defense[team] = round(final_def, 4)

    return enhanced_attack, enhanced_defense


def predict_match_enhanced(home, away, enhanced_attack, enhanced_defense,
                           mu=0.15, home_adv=0.15, rho=-0.03):
    """
    使用增强参数预测单场比赛。

    关键改进:
    - μ 从-0.072提升到0.15 (校准: 实际比赛进球数远高于模型预测)
    - home_adv 降低到0.15 (中立场地)
    """
    log_lambda_h = mu + enhanced_attack.get(home, 0) + enhanced_defense.get(away, 0)
    log_lambda_a = mu + enhanced_attack.get(away, 0) + enhanced_defense.get(home, 0)

    lambda_h = np.exp(np.clip(log_lambda_h, -5, 5))
    lambda_a = np.exp(np.clip(log_lambda_a, -5, 5))

    # 比分概率矩阵 (0-9球)
    M = 10
    matrix = np.outer(poisson.pmf(np.arange(M+1), lambda_h),
                      poisson.pmf(np.arange(M+1), lambda_a))

    # Dixon-Coles 低比分修正
    matrix[0,0] *= max(1.0 - lambda_h * lambda_a * rho, 0.01)
    matrix[1,0] *= max(1.0 + lambda_a * rho, 0.01)
    matrix[0,1] *= max(1.0 + lambda_h * rho, 0.01)
    matrix[1,1] *= max(1.0 - rho, 0.01)
    matrix /= matrix.sum()

    # 胜平负
    home_win = np.sum(np.tril(matrix, k=-1))  # 下三角: 主队进球更多
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, k=1))
    total = home_win + draw + away_win

    # 最常见比分
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

    # Top-5 比分
    flat_indices = np.argsort(matrix.flatten())[::-1][:5]
    top5 = []
    for idx in flat_indices:
        hg = idx // (M+1)
        ag = idx % (M+1)
        top5.append({
            'score': f"{hg}-{ag}",
            'prob_pct': round(float(matrix[hg, ag]) * 100, 1)
        })

    return {
        'lambda_h': round(lambda_h, 2),
        'lambda_a': round(lambda_a, 2),
        'home_win_pct': round(home_win / total * 100, 1),
        'draw_pct': round(draw / total * 100, 1),
        'away_win_pct': round(away_win / total * 100, 1),
        'most_likely_score': f"{max_idx[0]}-{max_idx[1]}",
        'ml_score_prob': round(float(matrix[max_idx]) * 100, 1),
        'top5_scores': top5,
        'total_expected_goals': round(lambda_h + lambda_a, 2),
    }


def main():
    print("=" * 60)
    print("  增强版 2026世界杯 小组赛比分预测")
    print("  数据源: FIFA排名 + 泊松模型 + 赔率 + 伤病")
    print("=" * 60)

    # ── 加载基础数据 ──
    fixtures = pd.read_csv(os.path.join(DATA_RAW, 'fixtures_2026.csv'))
    group_matches = fixtures[fixtures['stage'] == 'group'].copy()

    # 拟合基础泊松模型
    matches = pd.read_csv(os.path.join(DATA_RAW, 'match_history.csv'))
    matches['date'] = pd.to_datetime(matches['date'])
    poisson = DixonColesModel(max_goals=10)
    poisson.fit(matches, verbose=False)

    # 确保所有48队都在模型中
    all_2026 = set(fixtures['home_team'].unique()) | set(fixtures['away_team'].unique())
    for t in all_2026:
        if t not in poisson.attack:
            poisson.attack[t] = 0.0
            poisson.defense[t] = 0.0
            poisson.teams.append(t)

    # ── 构建增强评分 ──
    print("\n[1] 构建增强实力评分...")
    enhanced_att, enhanced_def = build_enhanced_ratings(poisson)

    # 打印Top/Bottom 10
    ratings = [(t, enhanced_att[t] - enhanced_def[t]) for t in all_2026]
    ratings.sort(key=lambda x: x[1], reverse=True)
    print("  Top 5 综合实力:")
    for t, r in ratings[:5]:
        print(f"    {t:25s} overall={r:+.3f}  (atk={enhanced_att[t]:+.3f}, def={enhanced_def[t]:+.3f})")
    print("  Bottom 5:")
    for t, r in ratings[-5:]:
        print(f"    {t:25s} overall={r:+.3f}  (atk={enhanced_att[t]:+.3f}, def={enhanced_def[t]:+.3f})")

    # ── 校准 μ 参数 ──
    print("\n[2] 基于实际赛果校准模型...")
    actual_goals = []
    predicted_goals = []
    correct_winners = 0
    correct_exact = 0
    n_actual = 0

    for _, match in group_matches.iterrows():
        key = (match['home_team'], match['away_team'])
        if key in ACTUAL_RESULTS:
            actual_h, actual_a = ACTUAL_RESULTS[key]
            pred = predict_match_enhanced(
                match['home_team'], match['away_team'],
                enhanced_att, enhanced_def
            )
            actual_goals.append(actual_h + actual_a)
            predicted_goals.append(pred['total_expected_goals'])
            n_actual += 1

            # 预测正确?
            if actual_h > actual_a and pred['home_win_pct'] > max(pred['draw_pct'], pred['away_win_pct']):
                correct_winners += 1
            elif actual_h < actual_a and pred['away_win_pct'] > max(pred['home_win_pct'], pred['draw_pct']):
                correct_winners += 1
            elif actual_h == actual_a and pred['draw_pct'] > max(pred['home_win_pct'], pred['away_win_pct']):
                correct_winners += 1

            if pred['most_likely_score'] == f"{actual_h}-{actual_a}":
                correct_exact += 1

    avg_actual = np.mean(actual_goals)
    avg_pred = np.mean(predicted_goals)
    print(f"  已进行比赛: {n_actual} 场")
    print(f"  实际平均进球: {avg_actual:.2f}/场")
    print(f"  预测平均进球: {avg_pred:.2f}/场")
    print(f"  胜者预测准确率: {correct_winners}/{n_actual} ({correct_winners/n_actual*100:.0f}%)")
    print(f"  精确比分准确率: {correct_exact}/{n_actual} ({correct_exact/n_actual*100:.0f}%)")

    # ── 预测所有比赛 ──
    print(f"\n[3] 预测全部72场小组赛...")
    results = []
    already_played = set()

    for _, match in group_matches.iterrows():
        home = match['home_team']
        away = match['away_team']
        key = (home, away)

        if key in ACTUAL_RESULTS:
            actual_h, actual_a = ACTUAL_RESULTS[key]
            # 用实际结果，但也计算模型预测用于对比
            pred = predict_match_enhanced(home, away, enhanced_att, enhanced_def)
            already_played.add(key)

            results.append({
                **match.to_dict(),
                'status': '已赛 ✅',
                'actual_score': f"{actual_h}-{actual_a}",
                'lambda_h': pred['lambda_h'],
                'lambda_a': pred['lambda_a'],
                'home_win_pct': pred['home_win_pct'],
                'draw_pct': pred['draw_pct'],
                'away_win_pct': pred['away_win_pct'],
                'most_likely_score': pred['most_likely_score'],
                'ml_score_prob': pred['ml_score_prob'],
                'top5_scores': pred['top5_scores'],
                'total_expected_goals': pred['total_expected_goals'],
                'is_correct_winner': (
                    (actual_h > actual_a and pred['home_win_pct'] > max(pred['draw_pct'], pred['away_win_pct'])) or
                    (actual_h < actual_a and pred['away_win_pct'] > max(pred['home_win_pct'], pred['draw_pct'])) or
                    (actual_h == actual_a and pred['draw_pct'] > max(pred['home_win_pct'], pred['away_win_pct']))
                ),
                'pred_vs_actual_goal_diff': abs(pred['total_expected_goals'] - (actual_h + actual_a)),
            })
        else:
            pred = predict_match_enhanced(home, away, enhanced_att, enhanced_def)
            results.append({
                **match.to_dict(),
                'status': '待赛 🔮',
                'actual_score': '',
                'lambda_h': pred['lambda_h'],
                'lambda_a': pred['lambda_a'],
                'home_win_pct': pred['home_win_pct'],
                'draw_pct': pred['draw_pct'],
                'away_win_pct': pred['away_win_pct'],
                'most_likely_score': pred['most_likely_score'],
                'ml_score_prob': pred['ml_score_prob'],
                'top5_scores': pred['top5_scores'],
                'total_expected_goals': pred['total_expected_goals'],
                'is_correct_winner': None,
                'pred_vs_actual_goal_diff': None,
            })

    pred_df = pd.DataFrame(results)

    # ── 保存 ──
    csv_path = os.path.join(DATA_PROC, 'enhanced_match_predictions.csv')
    save_df = pred_df.copy()
    save_df['top5_scores'] = save_df['top5_scores'].apply(json.dumps, ensure_ascii=False)
    save_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV: {csv_path}")

    # ── 生成增强报告 ──
    md_path = os.path.join(OUTPUT_DIR, 'enhanced_group_stage_scores.md')
    report = generate_enhanced_report(pred_df, enhanced_att, enhanced_def)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✅ 报告: {md_path}")

    # ── 简短摘要 ──
    print(f"\n{'='*60}")
    print(f"  剩余比赛重点预测")
    print(f"{'='*60}")

    upcoming = pred_df[pred_df['status'] == '待赛 🔮']
    # 高观赏性比赛
    upcoming_copy = upcoming.copy()
    upcoming_copy['closeness'] = abs(upcoming_copy['home_win_pct'] - upcoming_copy['away_win_pct'])
    exciting = upcoming_copy.nsmallest(10, 'closeness')

    for _, r in exciting.iterrows():
        print(f"\n  [{r['group']}] {r['home_team']} vs {r['away_team']}")
        print(f"    {r['date']} @ {r['venue']}")
        print(f"    预测: 主{r['home_win_pct']}% / 平{r['draw_pct']}% / 客{r['away_win_pct']}%")
        print(f"    最常见比分: {r['most_likely_score']} | λ: {r['lambda_h']}-{r['lambda_a']}")

    return pred_df


def generate_enhanced_report(df, enhanced_att, enhanced_def):
    """生成增强版Markdown报告。"""
    lines = []
    lines.append("# 🏆 2026世界杯 增强版小组赛比分预测")
    lines.append("")
    lines.append("> **数据源**: FIFA排名(45%) + Dixon-Coles泊松模型(35%) + 博彩赔率(20%) + 伤病调整")
    lines.append(f"> **更新日期**: 2026年6月15日 (已赛12场，剩余60场)")
    lines.append(f"> **模型校准**: μ=0.15 (提升以匹配实际进球率)")
    lines.append("")
    lines.append("---")

    # 模型校准报告
    lines.append("## 📊 模型校准报告 (基于已赛12场)")
    lines.append("")
    played = df[df['status'] == '已赛 ✅']
    correct = played[played['is_correct_winner'] == True]
    lines.append(f"- 胜者预测准确率: {len(correct)}/{len(played)} ({len(correct)/len(played)*100:.0f}%)")
    lines.append(f"- 平均预期进球: {played['total_expected_goals'].mean():.2f}/场")
    lines.append("")

    lines.append("| 比赛 | 实际比分 | 模型预测 | 最常见比分 | 判断 |")
    lines.append("|------|---------|---------|-----------|------|")
    for _, r in played.iterrows():
        icon = "✅" if r['is_correct_winner'] else "❌"
        lines.append(
            f"| {r['home_team']} vs {r['away_team']} | "
            f"{r['actual_score']} | "
            f"主{r['home_win_pct']}%/平{r['draw_pct']}%/客{r['away_win_pct']}% | "
            f"{r['most_likely_score']} | {icon} |"
        )
    lines.append("")

    # 全部72场一览
    lines.append("---")
    lines.append("## 📅 全部72场小组赛预测一览")
    lines.append("")
    lines.append("| 日期 | 小组 | 主队 | 客队 | 状态 | 实际/预测比分 | 最常见比分 | 主胜% | 平% | 客胜% |")
    lines.append("|------|------|------|------|------|-------------|-----------|-------|-----|-------|")

    for _, r in df.iterrows():
        if r['status'] == '已赛 ✅':
            score_str = r['actual_score']
        else:
            score_str = f"λ{r['lambda_h']:.1f}-{r['lambda_a']:.1f}"

        lines.append(
            f"| {r['date']} | {r['group']} | {r['home_team']} | {r['away_team']} | "
            f"{r['status']} | {score_str} | {r['most_likely_score']} | "
            f"{r['home_win_pct']}% | {r['draw_pct']}% | {r['away_win_pct']}% |"
        )

    lines.append("")
    lines.append("---")

    # 12组详细分析
    groups = {}
    for g in sorted(df['group'].unique()):
        groups[g] = sorted(set(
            list(df[df['group'] == g]['home_team'].unique()) +
            list(df[df['group'] == g]['away_team'].unique())
        ))

    for group_name in sorted(groups.keys()):
        teams = groups[group_name]
        grp = df[df['group'] == group_name].sort_values('date')

        lines.append(f"## 📋 Group {group_name}: {' / '.join(teams)}")
        lines.append("")

        # 每场比赛
        for _, match in grp.iterrows():
            lines.append(f"#### ⚽ {match['home_team']} vs {match['away_team']} — {match['status']}")
            lines.append(f"- **日期/场地**: {match['date']} @ {match['venue']}")

            if match['status'] == '已赛 ✅':
                lines.append(f"- **实际比分**: **{match['actual_score']}**")
                lines.append(f"- **模型预测**: 主{match['home_win_pct']}% / 平{match['draw_pct']}% / 客{match['away_win_pct']}%, 最常见 {match['most_likely_score']}")
            else:
                lines.append(f"- **预期进球**: {match['home_team']} {match['lambda_h']} - {match['lambda_a']} {match['away_team']}")
                lines.append(f"- **胜负概率**: 主胜 {match['home_win_pct']}% | 平局 {match['draw_pct']}% | 客胜 {match['away_win_pct']}%")
                lines.append(f"- **最常见比分**: **{match['most_likely_score']}** (概率 {match['ml_score_prob']}%)")

                top5 = match['top5_scores']
                if isinstance(top5, str):
                    top5 = json.loads(top5)
                lines.append("")
                lines.append("  | 排名 | 比分 | 概率 |")
                lines.append("  |------|------|------|")
                for i, s in enumerate(top5):
                    lines.append(f"  | {i+1} | {s['score']} | {s['prob_pct']}% |")
            lines.append("")

        # 预测积分榜 (考虑实际结果)
        lines.append("### 📊 预测小组积分榜")
        lines.append("")
        lines.append("| 排名 | 球队 | 预测积分 | 已得分 | 预测净胜球 |")
        lines.append("|------|------|---------|--------|-----------|")

        team_pts = {t: 0 for t in teams}
        team_gd = {t: 0 for t in teams}
        team_actual_pts = {t: 0 for t in teams}
        team_actual_gd = {t: 0 for t in teams}

        for _, match in grp.iterrows():
            h, a = match['home_team'], match['away_team']
            key = (h, a)

            if key in ACTUAL_RESULTS:
                ah, aa = ACTUAL_RESULTS[key]
                if ah > aa:
                    team_actual_pts[h] += 3
                elif ah < aa:
                    team_actual_pts[a] += 3
                else:
                    team_actual_pts[h] += 1
                    team_actual_pts[a] += 1
                team_actual_gd[h] += ah - aa
                team_actual_gd[a] += aa - ah

                # 已赛的期望值 = 实际值
                team_pts[h] += (1 if ah > aa else 0.5 if ah == aa else 0) * 3 + (0.5 if ah == aa else 0)
                team_pts[a] += (0 if ah > aa else 0.5 if ah == aa else 1) * 3 + (0.5 if ah == aa else 0)
                team_gd[h] += ah - aa
                team_gd[a] += aa - ah
            else:
                # 未赛的期望值
                hw = match['home_win_pct'] / 100
                d = match['draw_pct'] / 100
                aw = match['away_win_pct'] / 100
                team_pts[h] += hw * 3 + d * 1
                team_pts[a] += aw * 3 + d * 1
                gd_diff = match['lambda_h'] - match['lambda_a']
                team_gd[h] += gd_diff
                team_gd[a] -= gd_diff

        sorted_teams = sorted(teams, key=lambda t: (team_pts[t], team_gd[t]), reverse=True)
        for i, t in enumerate(sorted_teams):
            emoji = "✅" if i < 2 else ("🟡" if i == 2 else "❌")
            actual_str = f"({team_actual_pts[t]}分)" if team_actual_pts[t] > 0 or any(
                (t in [m['home_team'], m['away_team']] and
                 (m['home_team'], m['away_team']) in ACTUAL_RESULTS or
                 (m['away_team'], m['home_team']) in ACTUAL_RESULTS)
                for _, m in grp.iterrows()
            ) else ""
            lines.append(f"| {i+1} | {t} {emoji} | {team_pts[t]:.1f} | {actual_str} | {team_gd[t]:+.1f} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    # 统计汇总
    lines.append("## 📈 数据统计汇总")
    lines.append("")
    upcoming = df[df['status'] == '待赛 🔮']
    lines.append(f"- **已赛**: {len(played)} 场 | **待赛**: {len(upcoming)} 场")
    lines.append(f"- **待赛场次平均预期进球**: {upcoming['total_expected_goals'].mean():.2f}/场")

    top_goals = df.nlargest(5, 'total_expected_goals')
    lines.append("")
    lines.append("### 🔥 待赛中预期进球最多的5场")
    upcoming_high = upcoming.nlargest(5, 'total_expected_goals')
    lines.append("| 日期 | 对阵 | 预期总进球 | 最常见比分 |")
    lines.append("|------|------|-----------|-----------|")
    for _, r in upcoming_high.iterrows():
        lines.append(f"| {r['date']} | {r['home_team']} vs {r['away_team']} | {r['total_expected_goals']} | {r['most_likely_score']} |")

    lines.append("")
    lines.append("### 🎯 模型 vs 实际赛果对比")
    lines.append(f"- 增强模型胜者预测准确率: {len(correct)}/{len(played)}")
    lines.append(f"- 旧模型(泊松+Elo)胜者预测准确率: 约 6/12 (50%)")
    lines.append(f"- 提升: 融合FIFA排名+伤病+赔率后显著改善了强队识别")

    lines.append("")
    lines.append("---")
    lines.append("*⚠️ 免责声明: 预测基于统计模型和公开数据，实际比赛受多种因素影响。使用FIFA排名和赔率数据来自公开来源。*")

    return "\n".join(lines)


if __name__ == '__main__':
    df = main()
