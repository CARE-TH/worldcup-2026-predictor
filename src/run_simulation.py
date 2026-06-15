"""
2026世界杯蒙特卡洛模拟
使用已训练的模型进行 10,000 次完整赛事模拟
"""
import os, sys, json, pickle, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from poisson_model import DixonColesModel
from elo_calculator import EloRating
from xgboost_model import XGBoostMatchModel
from stacking_ensemble import StackingEnsemble
from calibration import ProbabilityCalibrator
from market_baseline import MarketBaseline
from data_pipeline import compute_recent_form
from match_predictor import MatchPredictor
from tournament_simulator import TournamentSimulator

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT, exist_ok=True)


def build_match_history_for_training():
    """加载全部可用历史数据并准备训练集。"""
    matches = pd.read_csv(os.path.join(DATA, 'raw', 'match_history.csv'))
    matches['date'] = pd.to_datetime(matches['date'])
    return matches


def train_final_models(matches):
    """用全部 534 场世界杯数据训练最终模型。"""
    print('[1] 训练最终模型（全部534场数据）...')

    # Poisson
    poisson = DixonColesModel()
    poisson.fit(matches, verbose=True)

    # Elo
    elo = EloRating()
    elo.fit_all(matches)

    # Recent form
    home_records = matches.rename(columns={'home_team': 'team', 'home_goals': 'goals_for', 'away_goals': 'goals_against'})
    away_records = matches.rename(columns={'away_team': 'team', 'away_goals': 'goals_for', 'home_goals': 'goals_against'})
    all_records = pd.concat([
        home_records[['date','team','goals_for','goals_against']],
        away_records[['date','team','goals_for','goals_against']]
    ])
    ref_date = pd.Timestamp('2026-06-01')
    recent_form = compute_recent_form(all_records, reference_date=ref_date)

    # XGBoost
    team_params_df = poisson.get_team_params_df()
    xgb = XGBoostMatchModel(n_estimators=300, max_depth=4)
    xgb_data = xgb.build_features(matches, team_params_df, elo.get_ratings_dict(), recent_form)
    X = xgb_data[xgb.feature_names]
    y = xgb_data['label']
    if len(X) > 0:
        xgb.fit(X, y, verbose=True)

    # Stacking
    stacking = StackingEnsemble()
    sample = matches.sample(min(200, len(matches)), random_state=42)
    base_train = {'poisson': [], 'elo': [], 'xgb': []}
    y_stack = []

    for _, match in sample.iterrows():
        try:
            pr = poisson.predict_result(match['home_team'], match['away_team'])
            base_train['poisson'].append([pr['away_win_pct']/100, pr['draw_pct']/100, pr['home_win_pct']/100])
        except:
            base_train['poisson'].append([1/3]*3)
        try:
            er = elo.predict(match['home_team'], match['away_team'])
            base_train['elo'].append([er['away_win_pct']/100, er['draw_pct']/100, er['home_win_pct']/100])
        except:
            base_train['elo'].append([1/3]*3)
        try:
            feats = xgb._build_single_match_features(match['home_team'], match['away_team'],
                                                       team_params_df, elo.get_ratings_dict(), recent_form)
            xr = xgb.predict_single(feats)
            base_train['xgb'].append([xr['away_win_pct']/100, xr['draw_pct']/100, xr['home_win_pct']/100])
        except:
            base_train['xgb'].append([1/3]*3)
        if match['home_goals'] > match['away_goals']: y_stack.append(2)
        elif match['home_goals'] == match['away_goals']: y_stack.append(1)
        else: y_stack.append(0)

    stacking.fit({k: np.array(v) for k,v in base_train.items()}, np.array(y_stack), verbose=True)

    # Calibrator
    cal_preds, cal_y = [], []
    for i, (_, match) in enumerate(sample.iterrows()):
        bp = {k: np.array([v[i]]) for k,v in base_train.items()}
        try:
            sr = stacking.predict_single(bp)
            cal_preds.append([sr['away_win_pct']/100, sr['draw_pct']/100, sr['home_win_pct']/100])
            if match['home_goals'] > match['away_goals']: cal_y.append(2)
            elif match['home_goals'] == match['away_goals']: cal_y.append(1)
            else: cal_y.append(0)
        except: pass

    calibrator = ProbabilityCalibrator(method='isotonic')
    if len(cal_preds) > 20:
        calibrator.fit(np.array(cal_preds), np.array(cal_y))

    # Team attributes
    team_attrs = pd.read_csv(os.path.join(DATA, 'raw', 'team_attributes.csv'))

    # 为无历史数据球队补默认参数
    fixtures = pd.read_csv(os.path.join(DATA, 'raw', 'fixtures_2026.csv'))
    all_teams = set(fixtures['home_team'].unique()) | set(fixtures['away_team'].unique())
    for t in all_teams:
        if t not in poisson.attack:
            poisson.attack[t] = 0.0
            poisson.defense[t] = 0.0
            poisson.teams.append(t)
        if t not in elo.ratings:
            row = team_attrs[team_attrs['team'] == t]
            if len(row) and not pd.isna(row['fifa_rank'].values[0]):
                elo.ratings[t] = max(1000, 1600 - (row['fifa_rank'].values[0] - 1) * 6)
            else:
                elo.ratings[t] = 1200

    # Market
    market = MarketBaseline()

    # Predictor
    predictor = MatchPredictor(
        poisson_model=poisson, elo=elo, xgb_model=xgb,
        stacking=stacking, calibrator=calibrator, market=market,
        team_attributes=team_attrs, recent_form=recent_form,
    )

    return predictor, poisson, elo


def main():
    matches = build_match_history_for_training()
    predictor, poisson, elo = train_final_models(matches)

    # 加载赛程
    fixtures = pd.read_csv(os.path.join(DATA, 'raw', 'fixtures_2026.csv'))

    print(f'\n[2] 2026世界杯模拟 (10000次)...')
    sim = TournamentSimulator(predictor, fixtures, n_simulations=10000, random_seed=42)
    t0 = time.time()
    results = sim.run(verbose=True)
    elapsed = time.time() - t0

    # 打印结果
    sim.print_summary(top_n=15)

    # 保存
    sim.save_results(os.path.join(OUTPUT, 'simulation_results.json'))

    # 导出球队参数供分析
    params_df = poisson.get_team_params_df()
    params_df.to_csv(os.path.join(OUTPUT, 'team_params_final.csv'), index=False)

    elo_df = elo.get_ratings_df()
    elo_df.to_csv(os.path.join(OUTPUT, 'elo_ratings_final.csv'), index=False)

    print(f'\n[OK] 模拟完成 ({elapsed:.0f}s)')
    print(f'结果文件: output/simulation_results.json')


if __name__ == '__main__':
    main()
