"""
小组赛比分预测脚本
================
使用 Dixon-Coles 泊松模型预测2026世界杯全部72场小组赛的比分。

输出:
- data/processed/group_match_predictions.csv  每场比赛的详细预测
- output/group_stage_scores.md               12组完整比分预测报告
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(__file__))

from poisson_model import DixonColesModel
from elo_calculator import EloRating

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 举办城市海拔（米），用于计算高原影响
VENUE_ALTITUDE = {
    "Mexico City": 2240, "Guadalajara": 1566, "Monterrey": 540,
    "Los Angeles": 71, "San Francisco": 16, "Seattle": 50,
    "Vancouver": 2, "New York": 10, "Boston": 43,
    "Philadelphia": 12, "Atlanta": 320, "Miami": 2,
    "Dallas": 131, "Houston": 13, "Kansas City": 277, "Toronto": 76,
}

# 2026世界杯主办国
HOST_NATIONS = {"United States", "Mexico", "Canada"}


def fit_poisson_model():
    """从历史数据拟合 Dixon-Coles 泊松模型。"""
    matches = pd.read_csv(os.path.join(DATA_RAW, 'match_history.csv'))
    matches['date'] = pd.to_datetime(matches['date'])

    if 'tournament_type' not in matches.columns:
        matches['tournament_type'] = 'FIFA World Cup'
    if 'is_knockout' not in matches.columns:
        matches['is_knockout'] = False
    if 'weight' not in matches.columns:
        matches['weight'] = 1.0

    model = DixonColesModel(max_goals=10)
    model.fit(matches, verbose=True)

    return model


def predict_group_matches(poisson_model, elo_model, fixtures):
    """
    预测全部小组赛比分。

    返回 DataFrame 包含每场的:
    - 预期进球 (λ_h, λ_a)
    - 胜/平/负概率
    - 最常见比分及概率
    - Top-5 比分概率
    - Elo 胜率
    """
    results = []

    for _, match in fixtures.iterrows():
        home = match['home_team']
        away = match['away_team']
        group = match['group']
        match_id = match['match_id']
        date = match['date']
        venue = match['venue']

        # 海拔加成（高海拔球队可能有优势，但这里简化处理）
        altitude = VENUE_ALTITUDE.get(venue, 0)

        # 主办国在中立场地也有一点主场氛围加成
        home_boost = 1.0
        if home in HOST_NATIONS and altitude > 0:
            home_boost = 1.05  # 轻微主场氛围

        # ── 泊松模型预测 ──
        try:
            poisson_pred = poisson_model.predict_result(home, away, is_neutral=True)
            lambda_h, lambda_a = poisson_model.expected_goals(home, away, is_neutral=True)
            score_matrix = poisson_model.score_probability_matrix(home, away, is_neutral=True)

            # 获取 Top-5 最可能的比分
            flat_indices = np.argsort(score_matrix.flatten())[::-1][:5]
            top5_scores = []
            for idx in flat_indices:
                h_goals = idx // score_matrix.shape[1]
                a_goals = idx % score_matrix.shape[1]
                prob = score_matrix[h_goals, a_goals] * 100
                top5_scores.append({
                    'score': f"{h_goals}-{a_goals}",
                    'prob_pct': round(prob, 1)
                })
        except KeyError as e:
            print(f"  ⚠️ 球队缺失: {e}, 使用默认值")
            lambda_h, lambda_a = 1.2, 0.9
            poisson_pred = {
                'home_win_pct': 38.0, 'draw_pct': 28.0, 'away_win_pct': 34.0,
                'expected_home_goals': 1.2, 'expected_away_goals': 0.9,
                'most_likely_score': '1-0', 'score_probability': 10.5
            }
            top5_scores = [
                {'score': '1-0', 'prob_pct': 10.5},
                {'score': '1-1', 'prob_pct': 9.5},
                {'score': '0-0', 'prob_pct': 8.5},
                {'score': '2-0', 'prob_pct': 7.5},
                {'score': '0-1', 'prob_pct': 7.0},
            ]

        # ── Elo 模型预测 ──
        try:
            elo_pred = elo_model.predict(home, away, is_neutral=True, with_draw=True)
            elo_h_win = elo_pred['home_win_pct']
            elo_draw = elo_pred['draw_pct']
            elo_a_win = elo_pred['away_win_pct']
            elo_diff = elo_pred.get('elo_diff', 0)
        except:
            elo_h_win, elo_draw, elo_a_win = 36, 28, 36
            elo_diff = 0

        # ── 综合预测 ──
        # 权重: 泊松 60% + Elo 40%（泊松对细粒度比分更准，Elo 对胜负趋势更稳）
        w_poisson, w_elo = 0.60, 0.40
        final_h = poisson_pred['home_win_pct'] * w_poisson + elo_h_win * w_elo
        final_d = poisson_pred['draw_pct'] * w_poisson + elo_draw * w_elo
        final_a = poisson_pred['away_win_pct'] * w_poisson + elo_a_win * w_elo

        # 归一化
        total = final_h + final_d + final_a
        final_h = final_h / total * 100
        final_d = final_d / total * 100
        final_a = final_a / total * 100

        # 置信度
        max_prob = max(final_h, final_d, final_a)
        poisson_elo_spread = abs(poisson_pred['home_win_pct'] - elo_h_win)
        if max_prob > 50 and poisson_elo_spread < 10:
            confidence = "高 ✅"
        elif max_prob > 38 or poisson_elo_spread < 18:
            confidence = "中 ⚠️"
        else:
            confidence = "低 ❓"

        # 评级该场比赛的观赏性
        # 双方实力越接近 + 预期进球越多 = 越好看
        closeness = 1.0 - abs(final_h - final_a) / 100  # 0~1
        total_goals = lambda_h + lambda_a
        spectacle = closeness * 0.6 + min(total_goals / 4.0, 1.0) * 0.4
        if spectacle > 0.7:
            stars = "⭐⭐⭐"
        elif spectacle > 0.5:
            stars = "⭐⭐"
        else:
            stars = "⭐"

        results.append({
            'group': group,
            'match_id': match_id,
            'date': date,
            'home_team': home,
            'away_team': away,
            'venue': venue,
            'lambda_h': round(lambda_h, 2),
            'lambda_a': round(lambda_a, 2),
            'home_win_pct': round(final_h, 1),
            'draw_pct': round(final_d, 1),
            'away_win_pct': round(final_a, 1),
            'most_likely_score': poisson_pred['most_likely_score'],
            'ml_score_prob': poisson_pred.get('score_probability', 0),
            'top5_scores': top5_scores,
            'elo_diff': round(elo_diff, 1),
            'total_expected_goals': round(lambda_h + lambda_a, 2),
            'confidence': confidence,
            'spectacle': stars,
        })

    return pd.DataFrame(results)


def generate_markdown_report(df, groups):
    """生成12组小组赛比分预测 Markdown 报告。"""

    report = []
    report.append("# 🏆 2026世界杯 — 小组赛比分预测报告")
    report.append("")
    report.append("> **预测方法**: Dixon-Coles 泊松模型 (60%) + Elo 评分模型 (40%) 加权融合")
    report.append("> **数据基准日**: 2026年6月1日")
    report.append("> **说明**: 所有比赛均为中立场地；预期进球基于攻防参数计算，最常见比分来自泊松概率矩阵")
    report.append("")
    report.append("---")

    # 全赛程总览
    report.append("## 📅 全部72场小组赛一览")
    report.append("")
    report.append("| 日期 | 小组 | 主队 | 客队 | 预期比分 | 最常见比分 | 主胜% | 平% | 客胜% | 观赏性 | 置信度 |")
    report.append("|------|------|------|------|----------|-----------|-------|-----|-------|--------|--------|")

    for _, row in df.iterrows():
        expected_score = f"{row['lambda_h']:.1f} - {row['lambda_a']:.1f}"
        report.append(
            f"| {row['date']} | {row['group']} | {row['home_team']} | {row['away_team']} | "
            f"{expected_score} | {row['most_likely_score']} | "
            f"{row['home_win_pct']}% | {row['draw_pct']}% | {row['away_win_pct']}% | "
            f"{row['spectacle']} | {row['confidence']} |"
        )

    report.append("")
    report.append("---")

    # 12组详细分析
    for group_name in sorted(groups.keys()):
        teams = groups[group_name]
        group_df = df[df['group'] == group_name].sort_values('match_id')

        report.append(f"## 📋 Group {group_name}: {' / '.join(teams)}")
        report.append("")

        # 该组各队参数
        report.append("### 🎯 每场比赛详细预测")
        report.append("")

        for _, match in group_df.iterrows():
            report.append(f"#### ⚽ {match['home_team']} vs {match['away_team']}")
            report.append(f"- **日期/场地**: {match['date']} @ {match['venue']}")
            report.append(f"- **预期进球**: {match['home_team']} {match['lambda_h']} - {match['lambda_a']} {match['away_team']}")
            report.append(f"- **胜负概率**: 主胜 {match['home_win_pct']}% | 平局 {match['draw_pct']}% | 客胜 {match['away_win_pct']}%")
            report.append(f"- **最常见比分**: **{match['most_likely_score']}** (概率 {match['ml_score_prob']}%)")
            report.append(f"- **置信度**: {match['confidence']} | 观赏性: {match['spectacle']}")
            report.append("")

            # Top 5 比分
            top5 = match['top5_scores']
            if isinstance(top5, str):
                top5 = eval(top5)
            report.append("  | 排名 | 比分 | 概率 |")
            report.append("  |------|------|------|")
            for i, s in enumerate(top5):
                report.append(f"  | {i+1} | {s['score']} | {s['prob_pct']}% |")
            report.append("")

        # 小组积分预测
        report.append("### 📊 预测小组积分榜")
        report.append("")
        report.append("| 排名 | 球队 | 预测积分 | 预测净胜球 |")
        report.append("|------|------|---------|-----------|")

        # 计算预测积分
        team_pts = {t: 0 for t in teams}
        team_gd = {t: 0 for t in teams}
        team_gf = {t: 0 for t in teams}

        for _, match in group_df.iterrows():
            h = match['home_team']
            a = match['away_team']
            h_win = match['home_win_pct'] / 100
            d = match['draw_pct'] / 100
            a_win = match['away_win_pct'] / 100

            # 期望积分
            team_pts[h] += h_win * 3 + d * 1
            team_pts[a] += a_win * 3 + d * 1

            # 期望净胜球
            exp_gd = match['lambda_h'] - match['lambda_a']
            team_gd[h] += exp_gd
            team_gd[a] -= exp_gd
            team_gf[h] += match['lambda_h']
            team_gf[a] += match['lambda_a']

        sorted_teams = sorted(teams, key=lambda t: (team_pts[t], team_gd[t], team_gf[t]), reverse=True)

        for i, t in enumerate(sorted_teams):
            emoji = "✅" if i < 2 else ("🟡" if i == 2 else "❌")
            report.append(f"| {i+1} | {t} {emoji} | {team_pts[t]:.1f} | {team_gd[t]:+.1f} |")

        report.append("")
        report.append("---")
        report.append("")

    # 汇总统计
    report.append("## 📈 数据统计汇总")
    report.append("")

    high_conf = len(df[df['confidence'].str.contains('高')])
    mid_conf = len(df[df['confidence'].str.contains('中')])
    low_conf = len(df[df['confidence'].str.contains('低')])

    report.append(f"- **总比赛数**: {len(df)} 场")
    report.append(f"- **置信度分布**: 高 {high_conf} 场 | 中 {mid_conf} 场 | 低 {low_conf} 场")
    report.append(f"- **平均预期进球/场**: {df['total_expected_goals'].mean():.2f}")

    # 最高/最低进球比赛
    top_goals = df.nlargest(5, 'total_expected_goals')
    report.append("")
    report.append("### 🔥 预期进球最多的5场比赛")
    report.append("| 日期 | 对阵 | 预期总进球 | 最常见比分 |")
    report.append("|------|------|-----------|-----------|")
    for _, r in top_goals.iterrows():
        report.append(f"| {r['date']} | {r['home_team']} vs {r['away_team']} | {r['total_expected_goals']} | {r['most_likely_score']} |")

    report.append("")
    report.append("### 🧊 最可能零封的比赛（最低进球预期）")
    low_goals = df.nsmallest(5, 'total_expected_goals')
    report.append("| 日期 | 对阵 | 预期总进球 | 最常见比分 |")
    report.append("|------|------|-----------|-----------|")
    for _, r in low_goals.iterrows():
        report.append(f"| {r['date']} | {r['home_team']} vs {r['away_team']} | {r['total_expected_goals']} | {r['most_likely_score']} |")

    report.append("")
    report.append("---")
    report.append("")
    report.append("*⚠️ 免责声明: 以上预测基于历史数据和统计模型，足球比赛结果受多种不可预测因素影响（天气、伤病、裁判、运气等），预测仅供参考。*")

    return "\n".join(report)


def main():
    print("=" * 60)
    print("  2026世界杯 — 小组赛比分预测")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1] 加载数据...")
    fixtures = pd.read_csv(os.path.join(DATA_RAW, 'fixtures_2026.csv'))
    group_matches = fixtures[fixtures['stage'] == 'group'].copy()
    print(f"  小组赛场次: {len(group_matches)}")

    # 提取分组信息
    groups = {}
    for g in sorted(group_matches['group'].unique()):
        teams = set(group_matches[group_matches['group'] == g]['home_team'].unique()) | \
                set(group_matches[group_matches['group'] == g]['away_team'].unique())
        groups[g] = sorted(teams)
    print(f"  小组数: {len(groups)}")

    # ── 2. 拟合泊松模型 ──
    print("\n[2] 拟合 Dixon-Coles 泊松模型...")
    poisson_model = fit_poisson_model()
    print(f"  μ={poisson_model.mu:.4f}, home_adv={poisson_model.home_advantage:.4f}, ρ={poisson_model.rho:.4f}")

    # 确保所有48支2026参赛队都在模型中
    all_2026_teams = set(fixtures['home_team'].unique()) | set(fixtures['away_team'].unique())
    missing = all_2026_teams - set(poisson_model.teams)
    if missing:
        print(f"  ⚠️ {len(missing)} 支球队无历史数据, 使用默认攻防参数:")
        for t in sorted(missing):
            poisson_model.attack[t] = 0.0
            poisson_model.defense[t] = 0.0
            poisson_model.teams.append(t)
            print(f"    - {t}")

    # ── 3. 加载 Elo ──
    print("\n[3] 加载 Elo 评分...")
    elo_model = EloRating()
    elo_df = pd.read_csv(os.path.join(DATA_PROC, 'elo_ratings.csv'))
    for _, row in elo_df.iterrows():
        elo_model.ratings[row['team']] = row['elo_rating']
    print(f"  已加载 {len(elo_model.ratings)} 支球队 Elo 评分")

    # ── 4. 预测 ──
    print("\n[4] 预测全部72场小组赛比分...")
    predictions_df = predict_group_matches(poisson_model, elo_model, group_matches)

    # 保存 CSV
    csv_path = os.path.join(DATA_PROC, 'group_match_predictions.csv')
    # 把 top5_scores 序列化
    save_df = predictions_df.copy()
    save_df['top5_scores'] = save_df['top5_scores'].apply(json.dumps, ensure_ascii=False)
    save_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV 已保存: {csv_path}")

    # ── 5. 生成 Markdown ──
    print("\n[5] 生成 Markdown 报告...")
    md_report = generate_markdown_report(predictions_df, groups)
    md_path = os.path.join(OUTPUT_DIR, 'group_stage_scores.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"  ✅ 报告已保存: {md_path}")

    # ── 6. 终端摘要 ──
    print("\n" + "=" * 60)
    print("  预测摘要 — 重点比赛")
    print("=" * 60)

    # 打印每组的焦点比赛
    for group_name in sorted(groups.keys()):
        grp = predictions_df[predictions_df['group'] == group_name]
        # 找实力最接近的比赛
        grp_copy = grp.copy()
        grp_copy['closeness'] = abs(grp_copy['home_win_pct'] - grp_copy['away_win_pct'])
        closest = grp_copy.nsmallest(1, 'closeness').iloc[0]

        print(f"\n  [Group {group_name}] {closest['home_team']} vs {closest['away_team']}")
        print(f"    {closest['date']} @ {closest['venue']}")
        print(f"    预测: 主{closest['home_win_pct']}% / 平{closest['draw_pct']}% / 客{closest['away_win_pct']}%")
        print(f"    最常见比分: {closest['most_likely_score']} | 预期进球: {closest['lambda_h']}-{closest['lambda_a']}")

    print(f"\n  完整报告请查看: {md_path}")
    print("=" * 60)

    return predictions_df


if __name__ == '__main__':
    df = main()
