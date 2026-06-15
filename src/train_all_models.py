"""
完整模型训练流水线
使用历史世界杯数据训练所有模型并保存结果。
"""
import os, sys, json, pickle, csv
import numpy as np
import pandas as pd
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import run_data_pipeline, classify_tournament
from poisson_model import DixonColesModel
from elo_calculator import EloRating
from xgboost_model import XGBoostMatchModel
from stacking_ensemble import StackingEnsemble
from calibration import ProbabilityCalibrator
from market_baseline import MarketBaseline

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW = os.path.join(DATA, 'raw')
PROC = os.path.join(DATA, 'processed')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(PROC, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

REFERENCE_DATE = pd.Timestamp('2026-06-01')  # 2026世界杯前夕


def main():
    print('=' * 55)
    print('  模型训练流水线')
    print(f'  基准日期: {REFERENCE_DATE.date()}')
    print('=' * 55)

    # ── 1. 加载数据 ──
    matches = pd.read_csv(os.path.join(RAW, 'match_history.csv'))
    matches['date'] = pd.to_datetime(matches['date'])
    matches['tournament_type'] = 'FIFA World Cup'
    matches['is_knockout'] = False  # 简化: 世界杯全标记为非淘汰赛先
    matches['weight'] = 1.0  # 等权

    team_attrs = pd.read_csv(os.path.join(RAW, 'team_attributes.csv'))
    fixtures = pd.read_csv(os.path.join(RAW, 'fixtures_2026.csv'))

    print(f'\n[1] 数据加载')
    print(f'  历史比赛: {len(matches)} 场 ({matches["date"].min().date()} ~ {matches["date"].max().date()})')
    print(f'  涉及球队: {matches["home_team"].nunique()} 支')
    print(f'  2026赛程: {len(fixtures)} 场')

    # ── 2. Dixon-Coles 泊松模型 ──
    print(f'\n[2] Dixon-Coles 泊松模型')

    # 映射: 需要确保所有48支2026参赛队都在模型中
    # 使用世界杯历史数据训练
    poisson = DixonColesModel()
    poisson.fit(matches, verbose=True)

    # 对不在历史数据中的球队, 用默认参数
    all_2026_teams = set(fixtures['home_team'].unique()) | set(fixtures['away_team'].unique())
    missing = all_2026_teams - set(poisson.teams)
    if missing:
        print(f'  ⚠️ {len(missing)} 支2026参赛队无历史数据, 使用默认参数:')
        for t in sorted(missing):
            if t not in poisson.attack:
                poisson.attack[t] = 0.0
                poisson.defense[t] = 0.0
                poisson.teams.append(t)
            print(f'    {t}')

    # 保存泊松参数
    team_params_df = poisson.get_team_params_df()
    team_params_df.to_csv(os.path.join(PROC, 'team_params.csv'), index=False)

    # ── 3. Elo 评分 ──
    print(f'\n[3] Elo 评分系统')
    elo = EloRating()
    elo.fit_all(matches)

    # 对无历史数据的球队设置默认Elo
    for t in all_2026_teams:
        if t not in elo.ratings:
            # 基于FIFA排名推断
            attr_row = team_attrs[team_attrs['team'] == t]
            if len(attr_row):
                rank = attr_row['fifa_rank'].values[0]
                # 排名 → 大致Elo: 1=1600, 50=1300, 100=1100
                inferred_elo = 1600 - (rank - 1) * 6
                elo.ratings[t] = max(1000, min(1600, inferred_elo))
            else:
                elo.ratings[t] = 1200

    elo_df = elo.get_ratings_df()
    elo_df.to_csv(os.path.join(PROC, 'elo_ratings.csv'), index=False)
    print(f'  {len(elo.ratings)} 支球队有Elo评分')
    print(f'  Top 5: {elo_df.head(5)[["team","elo_rating"]].to_dict("records")}')

    # ── 4. XGBoost 模型 ──
    print(f'\n[4] XGBoost 模型')
    xgb = XGBoostMatchModel(n_estimators=300, max_depth=4)

    # 从历史数据构建近期状态
    home_records = matches.rename(columns={'home_team': 'team', 'home_goals': 'goals_for', 'away_goals': 'goals_against'})
    away_records = matches.rename(columns={'away_team': 'team', 'away_goals': 'goals_for', 'home_goals': 'goals_against'})
    all_records = pd.concat([home_records[['date','team','goals_for','goals_against']],
                             away_records[['date','team','goals_for','goals_against']]])
    from data_pipeline import compute_recent_form
    recent_form = compute_recent_form(all_records, reference_date=REFERENCE_DATE)

    # 构建特征: 只用世界杯数据训练, 留最后一届做验证
    xgb_data = xgb.build_features(matches, team_params_df, elo.get_ratings_dict(), recent_form)
    X = xgb_data[xgb.feature_names]
    y = xgb_data['label']

    if len(X) > 0:
        xgb_metrics = xgb.fit(X, y, verbose=True)
        # 保存
        xgb.save(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
        # 特征重要性
        if xgb.feature_importance is not None:
            xgb.feature_importance.to_csv(os.path.join(PROC, 'feature_importance.csv'), index=False)
            print(f'  最重要特征: {xgb.feature_importance.head(5).to_dict("records")}')
    else:
        print('  ⚠️ 特征矩阵为空, 跳过XGBoost训练')

    # ── 5. Stacking 集成 ──
    print(f'\n[5] Stacking 集成')
    stacking = StackingEnsemble()

    # 收集各模型的训练集预测
    from collections import defaultdict
    base_preds_train = defaultdict(list)
    y_train_stacking = []

    # 采样训练
    sample = matches.sample(min(200, len(matches)), random_state=42)
    for _, match in sample.iterrows():
        try:
            pr = poisson.predict_result(match['home_team'], match['away_team'])
            base_preds_train['poisson'].append([pr['away_win_pct']/100, pr['draw_pct']/100, pr['home_win_pct']/100])
        except:
            base_preds_train['poisson'].append([0.33, 0.34, 0.33])

        try:
            er = elo.predict(match['home_team'], match['away_team'])
            base_preds_train['elo'].append([er['away_win_pct']/100, er['draw_pct']/100, er['home_win_pct']/100])
        except:
            base_preds_train['elo'].append([0.33, 0.34, 0.33])

        try:
            feats = xgb._build_single_match_features(
                match['home_team'], match['away_team'], team_params_df,
                elo.get_ratings_dict(), recent_form
            )
            xr = xgb.predict_single(feats)
            base_preds_train['xgb'].append([xr['away_win_pct']/100, xr['draw_pct']/100, xr['home_win_pct']/100])
        except:
            base_preds_train['xgb'].append([0.33, 0.34, 0.33])

        if match['home_goals'] > match['away_goals']:
            y_train_stacking.append(2)
        elif match['home_goals'] == match['away_goals']:
            y_train_stacking.append(1)
        else:
            y_train_stacking.append(0)

    base_preds_train_np = {k: np.array(v) for k, v in base_preds_train.items()}
    stacking_metrics = stacking.fit(base_preds_train_np, np.array(y_train_stacking), verbose=True)
    stacking.save(os.path.join(MODELS_DIR, 'stacking_meta.pkl'))

    # ── 6. 校准器 ──
    print(f'\n[6] 概率校准')
    calibrator = ProbabilityCalibrator(method='isotonic')

    # 收集Stacking后的训练预测
    all_preds = []
    all_y = []
    for i, (_, match) in enumerate(sample.iterrows()):
        bp = {k: np.array([v[i]]) for k, v in base_preds_train_np.items()}
        try:
            sr = stacking.predict_single(bp)
            all_preds.append([sr['away_win_pct']/100, sr['draw_pct']/100, sr['home_win_pct']/100])
            if match['home_goals'] > match['away_goals']: all_y.append(2)
            elif match['home_goals'] == match['away_goals']: all_y.append(1)
            else: all_y.append(0)
        except:
            pass

    if len(all_preds) > 20:
        calibrator.fit(np.array(all_preds), np.array(all_y))
        calibrator.save(os.path.join(MODELS_DIR, 'calibrator.pkl'))
    else:
        print('  ⚠️ 校准样本不足, 跳过')

    # ── 7. 最终汇总 ──
    print(f'\n{"="*55}')
    print(f'  训练完成！')
    print(f'{"="*55}')
    print(f'  模型文件:')
    for f in os.listdir(MODELS_DIR):
        print(f'    models/{f}')
    print(f'  处理后数据:')
    for f in os.listdir(PROC):
        print(f'    data/processed/{f}')


if __name__ == '__main__':
    main()
