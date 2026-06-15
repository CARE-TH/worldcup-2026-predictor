"""
回测6月14日比赛：增强模型预测 vs 实际结果
"""
import numpy as np
from scipy.stats import poisson

# ═══ 6月14日实际赛果 ═══
ACTUAL = [
    ('Germany', 'Curacao', 7, 1, 'E', 'Houston'),
    ('Cote d\'Ivoire', 'Ecuador', 1, 0, 'E', 'Philadelphia'),
    ('Netherlands', 'Japan', 2, 2, 'F', 'Dallas'),
    ('Sweden', 'Tunisia', 5, 1, 'F', 'Monterrey'),
]

FIFA = {
    'Germany':1744,'Curacao':1470,'Cote d\'Ivoire':1592,'Ecuador':1578,
    'Netherlands':1749,'Japan':1662,'Sweden':1610,'Tunisia':1530,
}

INJ = {
    'Germany':(-0.05,0.02),'Netherlands':(-0.15,0.10),'Japan':(-0.10,0.05),
}

# 6月14日赛前已知结果(6/11-13)
FORM = {'USA':0.08,'Mexico':0.05,'South Korea':0.05,'Australia':0.05}

def predict(home, away, mu=0.18):
    def r(t):
        f = FIFA.get(t, 1500)
        z = (f - 1580) / 180
        a = z * 0.45 + 0.05
        d = -z * 0.30 - 0.05
        ia, id_ = INJ.get(t, (0,0))
        a += ia + FORM.get(t, 0)
        d += id_
        return a, d

    ah, dh = r(home)
    aa, da = r(away)

    lh = np.exp(np.clip(0.18 + ah + da + 0.08, -5, 5))
    la = np.exp(np.clip(0.18 + aa + dh, -5, 5))

    M = 8
    m = np.outer(poisson.pmf(np.arange(M+1), lh), poisson.pmf(np.arange(M+1), la))
    rho = -0.02
    m[0,0]*=max(1-lh*la*rho,0.01); m[1,0]*=max(1+la*rho,0.01)
    m[0,1]*=max(1+lh*rho,0.01); m[1,1]*=max(1-rho,0.01)
    m/=m.sum()

    hw=np.sum(np.tril(m,k=-1)); dr=np.sum(np.diag(m)); aw=np.sum(np.triu(m,k=1))
    t=hw+dr+aw

    flat_idx = np.argsort(m.flatten())[::-1][:3]
    top3 = [(f'{idx//(M+1)}-{idx%(M+1)}', round(m[idx//(M+1),idx%(M+1)]*100,1)) for idx in flat_idx]
    mx = np.unravel_index(np.argmax(m), m.shape)

    return {
        'lh':round(lh,2),'la':round(la,2),
        'hw':round(hw/t*100,1),'d':round(dr/t*100,1),'aw':round(aw/t*100,1),
        'best':f'{mx[0]}-{mx[1]}','bp':round(m[mx]*100,1),
        'top3':top3,'tg':round(lh+la,2),
    }

print('=' * 90)
print('  算法回测: 6月14日比赛 -- 增强模型预测 vs 实际结果')
print('=' * 90)

correct_winner = 0
correct_exact = 0
total_goal_error = 0
brier_scores = []
total_abs_error = 0

for home, away, ah, aa, grp, venue in ACTUAL:
    p = predict(home, away)

    pred_winner = home if p['hw'] > max(p['d'], p['aw']) else (away if p['aw'] > max(p['hw'], p['d']) else 'draw')
    actual_winner = home if ah > aa else (away if aa > ah else 'draw')
    winner_ok = pred_winner == actual_winner
    exact_ok = p['best'] == f'{ah}-{aa}'

    if winner_ok: correct_winner += 1
    if exact_ok: correct_exact += 1
    total_goal_error += abs(p['tg'] - ah - aa)

    actual_probs = [1 if actual_winner == home else 0, 1 if actual_winner == 'draw' else 0, 1 if actual_winner == away else 0]
    pred_probs = [p['hw']/100, p['d']/100, p['aw']/100]
    brier = sum((a-pr)**2 for a,pr in zip(actual_probs, pred_probs))
    brier_scores.append(brier)

    print(f'\n┌{"─"*86}┐')
    print(f'│ [{grp}] {home} vs {away}  @ {venue}')
    print(f'├{"─"*86}┤')
    print(f'│ FIFA排名: {home}={FIFA.get(home,0):.0f}  {away}={FIFA.get(away,0):.0f}')

    h_bar = '█'*int(p['hw']/3); d_bar = '█'*int(p['d']/3); a_bar = '█'*int(p['aw']/3)
    print(f'│ 预测: 主{p["hw"]:5.1f}% {h_bar} | 平{p["d"]:5.1f}% {d_bar} | 客{p["aw"]:5.1f}% {a_bar}')
    print(f'│ 预期进球: {home} {p["lh"]:.1f} - {p["la"]:.1f} {away}  |  最常见: {p["best"]} ({p["bp"]}%)  |  总球预期: {p["tg"]:.1f}')
    print(f'│ Top3比分: {p["top3"][0][0]}={p["top3"][0][1]}%  {p["top3"][1][0]}={p["top3"][1][1]}%  {p["top3"][2][0]}={p["top3"][2][1]}%')

    print(f'├{"─"*86}┤')
    print(f'│ 实际比分: {home} {ah} - {aa} {away}  (总进球 {ah+aa})')
    w_icon = '✅' if winner_ok else '❌'
    s_icon = '✅' if exact_ok else '❌'
    print(f'│ 胜者判断: {w_icon}  精确比分: {s_icon}  总球误差: {abs(p["tg"]-ah-aa):.1f}  Brier: {brier:.3f}')

    # 分析
    if not winner_ok:
        print(f'│ ⚠️ 分析: ', end='')
        if home == 'Netherlands' and away == 'Japan':
            print('荷兰伤病严重(Simons/Schouten/de Ligt/Timber缺阵) + 日本韧性被低估')
        elif home == 'Sweden' and away == 'Tunisia':
            print('瑞典攻击力被模型低估，实际5球远超预期')
        elif home == 'Germany' and away == 'Curacao':
            print('Curacao防守被严重高估(FIFA排名差距=274分，预期7球屠杀合理但模型只给1.8球)')

# ═══ 汇总 ═══
n = len(ACTUAL)
print(f'\n{"="*90}')
print(f'  回测汇总')
print(f'{"="*90}')
print(f'  测试场次: {n}')
print(f'  胜者预测准确率: {correct_winner}/{n} = {correct_winner/n*100:.0f}%')
print(f'  精确比分准确率: {correct_exact}/{n} = {correct_exact/n*100:.0f}%')
print(f'  平均总进球误差: {total_goal_error/n:.2f}球/场')
print(f'  平均Brier Score: {np.mean(brier_scores):.3f} (0=完美, 1=最差)')

print(f'\n  逐场:')
for home, away, ah, aa, grp, venue in ACTUAL:
    p = predict(home, away)
    pred_fav = home if p['hw'] > max(p['d'],p['aw']) else (away if p['aw'] > max(p['hw'],p['d']) else 'draw')
    pred_score = f'{p["lh"]:.1f}-{p["la"]:.1f}'
    actual_score = f'{ah}-{aa}'
    ok = '✅' if pred_fav == (home if ah>aa else (away if aa>ah else 'draw')) else '❌'
    print(f'    [{grp}] {home} vs {away}: 预测{pred_score} → 实际{actual_score} (总球{p["tg"]:.1f}vs{ah+aa}) {ok}')

print(f'\n{"="*90}')
print(f'  关键发现:')
print(f'  1. 胜者预测 75%准确 -- 方向性判断基本可靠')
avg_actual = sum(ah+aa for _,_,ah,aa,_,_ in ACTUAL) / n
avg_pred = sum(predict(h,a)['tg'] for h,a,_,_,_,_ in ACTUAL) / n
print(f'  2. 总进球系统性低估 -- 实际平均{avg_actual:.1f}球 vs 预测平均{avg_pred:.1f}球')
print(f'  3. 强弱悬殊场次(德国vs库拉索)的屠杀级比分无法捕捉')
print(f'  4. 荷兰伤病(Simons/Schouten/de Ligt/Timber全缺)需要更大惩罚')
print(f'{"="*90}')
