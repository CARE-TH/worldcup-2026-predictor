"""
回测验证脚本：用历史数据验证模型预测能力
时间点一致性：只用世界杯之前的比赛训练，预测世界杯比赛
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from poisson_model import DixonColesModel
from elo_calculator import EloRating
from xgboost_model import XGBoostMatchModel
from stacking_ensemble import StackingEnsemble
from calibration import ProbabilityCalibrator
from market_baseline import MarketBaseline
from data_pipeline import compute_recent_form

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW = os.path.join(DATA, 'raw')
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT, exist_ok=True)


def backtest_year(matches_all, year, cutoff_date_str, verbose=True):
    """
    回测一届世界杯。

    Parameters
    ----------
    matches_all : pd.DataFrame
        全部历史比赛
    year : int
        世界杯年份
    cutoff_date_str : str
        训练截止日期 (世界杯开始前)
    """
    cutoff = pd.Timestamp(cutoff_date_str)
    wc_matches = matches_all[matches_all['date'].dt.year == year].copy()

    if len(wc_matches) == 0:
        print(f'  [SKIP] {year}: 无该届世界杯数据')
        return None

    # 只用世界杯前的数据训练
    train = matches_all[matches_all['date'] < cutoff].copy()
    print(f'\n{"="*55}')
    print(f'  {year} 世界杯回测')
    print(f'  训练截止: {cutoff.date()}')
    print(f'  训练数据: {len(train)} 场')
    print(f'  测试数据: {len(wc_matches)} 场')
    print(f'{"="*55}')

    # --- 训练泊松 ---
    poisson = DixonColesModel()
    poisson.fit(train, verbose=False)

    # --- 训练 Elo ---
    elo = EloRating()
    elo.fit_all(train)

    # --- 训练 XGBoost ---
    team_params_df = poisson.get_team_params_df()
    home_records = train.rename(columns={'home_team': 'team', 'home_goals': 'goals_for', 'away_goals': 'goals_against'})
    away_records = train.rename(columns={'away_team': 'team', 'away_goals': 'goals_for', 'home_goals': 'goals_against'})
    all_records = pd.concat([
        home_records[['date','team','goals_for','goals_against']],
        away_records[['date','team','goals_for','goals_against']]
    ])
    recent_form = compute_recent_form(all_records, reference_date=cutoff)

    xgb_model = XGBoostMatchModel(n_estimators=200, max_depth=4)
    xgb_data = xgb_model.build_features(train, team_params_df, elo.get_ratings_dict(), recent_form)
    X_train = xgb_data[xgb_model.feature_names]
    y_train = xgb_data['label']

    if len(X_train) > 20 and len(xgb_model.feature_names) > 0:
        xgb_model.fit(X_train, y_train, verbose=False)
    else:
        xgb_model._fitted = False  # skip XGBoost

    # --- 训练 Stacking ---
    stacking = StackingEnsemble()
    sample_n = min(150, len(train))
    sample_train = train.sample(sample_n, random_state=42)

    base_preds_train = {'poisson': [], 'elo': [], 'xgb': []}
    y_stacking = []

    for _, match in sample_train.iterrows():
        try:
            pr = poisson.predict_result(match['home_team'], match['away_team'])
            base_preds_train['poisson'].append([pr['away_win_pct']/100, pr['draw_pct']/100, pr['home_win_pct']/100])
        except:
            base_preds_train['poisson'].append([1/3]*3)

        try:
            er = elo.predict(match['home_team'], match['away_team'])
            base_preds_train['elo'].append([er['away_win_pct']/100, er['draw_pct']/100, er['home_win_pct']/100])
        except:
            base_preds_train['elo'].append([1/3]*3)

        try:
            feats = xgb_model._build_single_match_features(
                match['home_team'], match['away_team'], team_params_df,
                elo.get_ratings_dict(), recent_form
            )
            xr = xgb_model.predict_single(feats)
            base_preds_train['xgb'].append([xr['away_win_pct']/100, xr['draw_pct']/100, xr['home_win_pct']/100])
        except:
            base_preds_train['xgb'].append([1/3]*3)

        if match['home_goals'] > match['away_goals']: y_stacking.append(2)
        elif match['home_goals'] == match['away_goals']: y_stacking.append(1)
        else: y_stacking.append(0)

    base_preds_np = {k: np.array(v) for k,v in base_preds_train.items() if len(v) > 0}
    if len(base_preds_np) >= 2 and len(y_stacking) > 20:
        stacking.fit(base_preds_np, np.array(y_stacking), verbose=False)
    else:
        stacking._fitted = False

    # --- 预测该届世界杯每场比赛 ---
    predictions = []
    actuals = []

    for _, match in wc_matches.iterrows():
        h, a = match['home_team'], match['away_team']
        hg, ag = int(match['home_goals']), int(match['away_goals'])

        # 各模型预测
        try:
            pr = poisson.predict_result(h, a)
        except:
            pr = {'home_win_pct': 33.3, 'draw_pct': 33.4, 'away_win_pct': 33.3}

        try:
            er = elo.predict(h, a)
        except:
            er = {'home_win_pct': 33.3, 'draw_pct': 33.4, 'away_win_pct': 33.3}

        try:
            feats = xgb_model._build_single_match_features(h, a, team_params_df, elo.get_ratings_dict(), recent_form)
            xr = xgb_model.predict_single(feats)
        except:
            xr = {'home_win_pct': 33.3, 'draw_pct': 33.4, 'away_win_pct': 33.3}

        # 集成
        bp = {
            'poisson': np.array([pr['away_win_pct']/100, pr['draw_pct']/100, pr['home_win_pct']/100]),
            'elo': np.array([er['away_win_pct']/100, er['draw_pct']/100, er['home_win_pct']/100]),
        }
        if xgb_model._fitted:
            bp['xgb'] = np.array([xr['away_win_pct']/100, xr['draw_pct']/100, xr['home_win_pct']/100])

        if stacking._fitted:
            try:
                sr = stacking.predict_single(bp)
                final_probs = np.array([sr['away_win_pct']/100, sr['draw_pct']/100, sr['home_win_pct']/100])
            except:
                all_probs = np.array(list(bp.values()))
                final_probs = all_probs.mean(axis=0)
        else:
            all_probs = np.array(list(bp.values()))
            final_probs = all_probs.mean(axis=0)

        predictions.append(final_probs)

        if hg > ag: actuals.append(2)
        elif hg == ag: actuals.append(1)
        else: actuals.append(0)

    # --- 计算指标 ---
    pred_arr = np.array(predictions)
    actual_arr = np.array(actuals)
    pred_labels = np.argmax(pred_arr, axis=1)

    acc = (pred_labels == actual_arr).mean() * 100

    # Brier Score
    actual_onehot = np.eye(3)[actual_arr]
    brier = np.mean(np.sum((pred_arr - actual_onehot)**2, axis=1))

    # LogLoss
    eps = 1e-10
    logloss = -np.mean(np.log(np.clip(pred_arr[np.arange(len(actual_arr)), actual_arr], eps, 1.0)))

    # 方向准确率（忽略平局）
    direction_mask = actual_arr != 1
    if direction_mask.sum() > 0:
        direction_acc = (pred_labels[direction_mask] == actual_arr[direction_mask]).mean() * 100
    else:
        direction_acc = 0.0

    # 按阶段拆分
    if 'stage' in wc_matches.columns:
        pass  # could split by stage

    if verbose:
        print(f'\n  [回测结果]')
        print(f'  准确率:     {acc:.1f}%')
        print(f'  方向准确率: {direction_acc:.1f}% (忽略平局)')
        print(f'  Brier Score: {brier:.4f}')
        print(f'  LogLoss:     {logloss:.4f}')

    return {
        'year': year,
        'n_train': len(train), 'n_test': len(wc_matches),
        'accuracy': round(acc, 1), 'direction_accuracy': round(direction_acc, 1),
        'brier_score': round(brier, 4), 'log_loss': round(logloss, 4),
        'predictions': [p.tolist() for p in predictions],
        'actuals': actuals,
    }


def main():
    matches = pd.read_csv(os.path.join(RAW, 'match_history.csv'))
    matches['date'] = pd.to_datetime(matches['date'])

    results = {}

    # 2018 回测: 用 2018年6月前的数据预测 2018世界杯
    r2018 = backtest_year(matches, 2018, '2018-06-01')
    if r2018: results[2018] = r2018

    # 2022 回测: 用 2022年11月前的数据预测 2022世界杯
    r2022 = backtest_year(matches, 2022, '2022-11-01')
    if r2022: results[2022] = r2022

    # 总结
    print(f'\n{"="*55}')
    print(f'  回测总结')
    print(f'{"="*55}')
    for year, r in results.items():
        print(f'  {year}: Acc={r["accuracy"]}%  Brier={r["brier_score"]}  LogLoss={r["log_loss"]}  (n={r["n_test"]})')

    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    avg_brier = np.mean([r['brier_score'] for r in results.values()])
    print(f'  平均: Acc={avg_acc:.1f}%  Brier={avg_brier:.4f}')
    print(f'  基线: 随机猜测 Acc=33.3%  Brier=0.2222')

    # 保存
    with open(os.path.join(OUTPUT, 'backtest_results.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'predictions'}
                   for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f'\n[OK] 结果已保存到 output/backtest_results.json')


if __name__ == '__main__':
    main()
