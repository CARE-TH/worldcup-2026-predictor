"""
从搜索结果构建原始数据 CSV 文件。
"""
import csv, os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════
# 1. 赛程表 fixtures_2026.csv
# ═══════════════════════════════════════
fixtures = []

groups = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czechia'],
    'B': ['Canada', 'Bosnia & Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkiye'],
    'E': ['Germany', "Cote d'Ivoire", 'Ecuador', 'Curacao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'IR Iran', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cabo Verde'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'Congo DR'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

match_id = 1

# Matchday 1: team1 vs team3, team2 vs team4
md1_dates = {
    'A': '2026-06-11', 'B': '2026-06-12', 'C': '2026-06-13', 'D': '2026-06-12',
    'E': '2026-06-14', 'F': '2026-06-14', 'G': '2026-06-15', 'H': '2026-06-15',
    'I': '2026-06-16', 'J': '2026-06-16', 'K': '2026-06-17', 'L': '2026-06-17',
}
md1_venues = {
    'A': ['Mexico City', 'Guadalajara'], 'B': ['Toronto', 'San Francisco'],
    'C': ['New York', 'Boston'], 'D': ['Los Angeles', 'Vancouver'],
    'E': ['Houston', 'Philadelphia'], 'F': ['Dallas', 'Monterrey'],
    'G': ['Seattle', 'Los Angeles'], 'H': ['Atlanta', 'Miami'],
    'I': ['New York', 'Boston'], 'J': ['Kansas City', 'San Francisco'],
    'K': ['Houston', 'Mexico City'], 'L': ['Dallas', 'Toronto'],
}
md2_start = {'A':18,'B':18,'C':19,'D':19,'E':20,'F':20,'G':21,'H':21,'I':22,'J':22,'K':23,'L':23}
md3_start = {'A':24,'B':24,'C':24,'D':25,'E':25,'F':25,'G':26,'H':26,'I':26,'J':27,'K':27,'L':27}
md2_venues = {
    'A': ['Atlanta', 'Los Angeles'], 'B': ['Los Angeles', 'Vancouver'],
    'C': ['Boston', 'Philadelphia'], 'D': ['Seattle', 'San Francisco'],
    'E': ['Toronto', 'Kansas City'], 'F': ['Houston', 'Monterrey'],
    'G': ['Los Angeles', 'Vancouver'], 'H': ['Atlanta', 'Miami'],
    'I': ['Philadelphia', 'New York'], 'J': ['Dallas', 'San Francisco'],
    'K': ['Houston', 'Guadalajara'], 'L': ['Boston', 'Toronto'],
}
md3_venues = {
    'A': ['Mexico City', 'Monterrey'], 'B': ['Vancouver', 'Seattle'],
    'C': ['Miami', 'Atlanta'], 'D': ['Los Angeles', 'San Francisco'],
    'E': ['Philadelphia', 'New York'], 'F': ['Dallas', 'Kansas City'],
    'G': ['Seattle', 'Vancouver'], 'H': ['Houston', 'Guadalajara'],
    'I': ['Boston', 'Toronto'], 'J': ['Kansas City', 'Dallas'],
    'K': ['Miami', 'Atlanta'], 'L': ['New York', 'Philadelphia'],
}

for g in 'ABCDEFGHIJKL':
    t = groups[g]
    # MD1: 1-3, 2-4
    fixtures.append([g, match_id, md1_dates[g], t[0], t[2], md1_venues[g][0], 'group']); match_id += 1
    fixtures.append([g, match_id, md1_dates[g], t[1], t[3], md1_venues[g][1], 'group']); match_id += 1
    # MD2: 3-1, 4-2
    d2 = md2_start[g]
    fixtures.append([g, match_id, f'2026-06-{d2:02d}', t[2], t[0], md2_venues[g][0], 'group']); match_id += 1
    fixtures.append([g, match_id, f'2026-06-{d2:02d}', t[3], t[1], md2_venues[g][1], 'group']); match_id += 1
    # MD3: 2-3, 4-1
    d3 = md3_start[g]
    fixtures.append([g, match_id, f'2026-06-{d3:02d}', t[1], t[2], md3_venues[g][0], 'group']); match_id += 1
    fixtures.append([g, match_id, f'2026-06-{d3:02d}', t[3], t[0], md3_venues[g][1], 'group']); match_id += 1

with open(os.path.join(DATA_DIR, 'fixtures_2026.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group','match_id','date','home_team','away_team','venue','stage'])
    w.writerows(fixtures)
print(f'[OK] 赛程: {len(fixtures)} 场小组赛')

# ═══════════════════════════════════════
# 2. 球队属性 team_attributes.csv
# ═══════════════════════════════════════
team_data = [
    ['Argentina',    1, 0.940, 28.5, 19, 'Champion (5x)', 'Lionel Scaloni', 'Enzo Fernandez', 'Possession'],
    ['Spain',        2, 1.449, 26.7, 17, 'Champion (2010)', 'Luis de la Fuente', 'Lamine Yamal', 'Possession'],
    ['France',       3, 1.760, 27.1, 17, 'Champion (2x)', 'Didier Deschamps', 'Kylian Mbappe', 'Counter'],
    ['England',      4, 1.507, 26.9, 17, 'Champion (1966)', 'Thomas Tuchel', 'Jude Bellingham', 'Hybrid'],
    ['Portugal',     5, 1.173, 27.8,  9, 'Third (1966)', 'Roberto Martinez', 'Vitinha', 'Possession'],
    ['Brazil',       6, 1.049, 27.3, 23, 'Champion (5x)', 'Carlo Ancelotti', 'Vinicius Jr.', 'Attacking'],
    ['Morocco',      7, 0.561, 27.5,  7, 'Fourth (2022)', 'Walid Regragui', 'Achraf Hakimi', 'Counter'],
    ['Netherlands',  8, 0.963, 26.5, 12, 'Runner-up (3x)', 'Ronald Koeman', 'Ryan Gravenberch', 'Possession'],
    ['Belgium',      9, 0.624, 28.4, 15, 'Third (2018)', 'Domenico Tedesco', 'Jeremy Doku', 'Attacking'],
    ['Germany',     10, 1.148, 27.0, 21, 'Champion (4x)', 'Julian Nagelsmann', 'Jamal Musiala', 'Possession'],
    ['Croatia',     11, 0.448, 28.8,  7, 'Runner-up (2018)', 'Zlatko Dalic', 'Josko Gvardiol', 'Hybrid'],
    ['Colombia',    13, 0.350, 27.6,  7, 'Quarter-final (2014)', 'Nestor Lorenzo', 'Luis Diaz', 'Attacking'],
    ['Mexico',      14, 0.221, 28.2, 18, 'Quarter-final (2x)', 'Javier Aguirre', 'Santiago Gimenez', 'Counter'],
    ['Senegal',     15, 0.544, 26.3,  4, 'Quarter-final (2002)', 'Pape Thiaw', 'Sadio Mane', 'Physical'],
    ['Uruguay',     16, 0.466, 28.4, 15, 'Champion (2x)', 'Marcelo Bielsa', 'Federico Valverde', 'Hybrid'],
    ['United States',17, 0.439, 26.1, 12, 'Third (1930)', 'Mauricio Pochettino', 'Christian Pulisic', 'Counter'],
    ['Japan',       18, 0.321, 27.3,  8, 'R16 (4x)', 'Hajime Moriyasu', 'Takefusa Kubo', 'Possession'],
    ['Switzerland', 19, 0.382, 27.7, 13, 'Quarter-final (3x)', 'Murat Yakin', 'Granit Xhaka', 'Defensive'],
    ['IR Iran',     20, 0.037, 28.1,  7, 'Group stage', 'Amir Ghalenoei', 'Mehdi Taremi', 'Defensive'],
    ['Turkiye',     22, 0.544, 26.4,  3, 'Third (2002)', 'Vincenzo Montella', 'Arda Guler', 'Attacking'],
    ['Ecuador',     23, 0.433, 26.2,  5, 'R16 (2006)', 'Sebastian Beccacece', 'Moises Caicedo', 'Hybrid'],
    ['Austria',     24, 0.279, 27.0,  8, 'Third (1954)', 'Ralf Rangnick', 'David Alaba', 'Pressing'],
    ['South Korea', 25, 0.160, 27.8, 12, 'Fourth (2002)', 'Hong Myung-bo', 'Son Heung-min', 'Counter'],
    ['Australia',   27, 0.089, 27.5,  7, 'R16 (2x)', 'Tony Popovic', 'Nestory Irankunda', 'Physical'],
    ['Algeria',     28, 0.295, 26.8,  5, 'R16 (2014)', 'Vladimir Petkovic', 'Riyad Mahrez', 'Counter'],
    ['Egypt',       29, 0.134, 27.4,  4, 'Group stage', 'Hossam Hassan', 'Mohamed Salah', 'Counter'],
    ['Canada',      30, 0.226, 26.5,  3, 'Group stage', 'Jesse Marsch', 'Alphonso Davies', 'Pressing'],
    ['Norway',      31, 0.691, 25.9,  4, 'R16 (2x)', 'Stale Solbakken', 'Erling Haaland', 'Attacking'],
    ['Cote d\'Ivoire',33,0.611,25.8,  4,'Group stage', 'Emerse Fae', 'Simon Adingra', 'Attacking'],
    ['Panama',      34, 0.040, 30.4,  2, 'Group stage', 'Thomas Christiansen', 'Michael Murillo', 'Defensive'],
    ['Sweden',      38, 0.492, 27.3, 13, 'Runner-up (1958)', 'Jon Dahl Tomasson', 'Alexander Isak', 'Hybrid'],
    ['Czechia',     40, 0.216, 27.5, 10, 'Runner-up (2x)', 'Ivan Hasek', 'Patrik Schick', 'Counter'],
    ['Paraguay',    41, 0.177, 28.0,  9, 'Quarter-final (2010)', 'Gustavo Alfaro', 'Miguel Almiron', 'Defensive'],
    ['Scotland',    42, 0.196, 27.2,  9, 'Group stage', 'Steve Clarke', 'Andy Robertson', 'Physical'],
    ['Tunisia',     45, 0.080, 27.6,  7, 'Group stage', 'Jalel Kadri', 'Hannibal Mejbri', 'Defensive'],
    ['Congo DR',    46, 0.165, 26.9,  1, 'Group stage (1974)', 'Sebastien Desabre', 'Noah Sadiki', 'Physical'],
    ['Uzbekistan',  50, 0.098, 26.3,  1, 'Debut', 'Srecko Katanec', 'Eldor Shomurodov', 'Defensive'],
    ['Bosnia & Herzegovina',52,0.174,27.8,1,'Group stage (2014)', 'Sergej Barbarez', 'Edin Dzeko', 'Physical'],
    ['Ghana',       55, 0.270, 26.5,  5, 'Quarter-final (2010)', 'Otto Addo', 'Mohammed Kudus', 'Physical'],
    ['South Africa',59, 0.057, 27.0,  4, 'Group stage', 'Hugo Broos', 'Percy Tau', 'Counter'],
    ['Haiti',       62, 0.064, 28.2,  1, 'Group stage (1974)', 'Sebastien Migne', 'Frantzdy Pierrot', 'Physical'],
    ['Cabo Verde',  65, 0.063, 26.8,  1, 'Debut', 'Bubista', 'Ryan Mendes', 'Counter'],
    ['Saudi Arabia',67, 0.047, 27.5,  7, 'R16 (1994)', 'Roberto Mancini', 'Salem Al-Dawsari', 'Defensive'],
    ['Qatar',       72, 0.023, 27.3,  2, 'Group stage (2022)', 'Tintin Marquez', 'Akram Afif', 'Possession'],
    ['Iraq',        75, 0.024, 26.5,  2, 'Group stage (1986)', 'Jesus Casas', 'Ali Al-Hamadi', 'Counter'],
    ['New Zealand', 85, 0.040, 27.1,  3, 'Group stage', 'Darren Bazeley', 'Chris Wood', 'Physical'],
    ['Jordan',      88, 0.024, 26.7,  1, 'Debut', 'Hussein Ammouta', 'Mousa Al-Tamari', 'Defensive'],
    ['Curacao',     91, 0.030, 27.3,  1, 'Debut', 'Dean Gorre', 'Rangelo Janga', 'Counter'],
]

with open(os.path.join(DATA_DIR, 'team_attributes.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['team','fifa_rank','squad_value_billion','avg_age','world_cup_appearances',
                'best_result','coach','star_player','playing_style'])
    w.writerows(team_data)
print(f'[OK] 球队属性: {len(team_data)} 支')

# ═══════════════════════════════════════
# 3. 博彩赔率 betting_odds.csv
# ═══════════════════════════════════════
odds_list = [
    ['Spain',      5.50,  6.00],
    ['France',     5.70,  6.00],
    ['England',    7.50,  8.00],
    ['Brazil',     9.00, 10.00],
    ['Portugal',   9.00, 11.00],
    ['Argentina', 10.00, 11.00],
    ['Germany',   14.00, 15.00],
    ['Netherlands',17.00, 23.00],
    ['Belgium',   23.00, 41.00],
    ['Norway',    34.00, 36.00],
    ['Colombia',  36.00, 41.00],
    ['United States',51.00, 61.00],
    ['Morocco',   51.00, 67.00],
    ['Uruguay',   51.00, 67.00],
    ['Japan',     41.00, 66.00],
    ['Croatia',   66.00, 91.00],
    ['Senegal',   67.00, 81.00],
    ['Ecuador',   67.00, 101.00],
    ['Switzerland',81.00, 126.00],
    ['Turkiye',   81.00, 126.00],
    ['Austria',  101.00, 151.00],
    ["Cote d'Ivoire",101.00, 151.00],
    ['Sweden',   101.00, 151.00],
    ['Mexico',    61.00, 91.00],
    ['South Korea',126.00, 201.00],
    ['Australia', 151.00, 251.00],
    ['Canada',    176.00, 201.00],
    ['Egypt',     201.00, 301.00],
    ['Algeria',   151.00, 251.00],
    ['Ghana',     201.00, 301.00],
    ['Czechia',   201.00, 301.00],
    ['Scotland',  301.00, 501.00],
    ['Paraguay',  201.00, 301.00],
    ['IR Iran',   301.00, 501.00],
    ['Tunisia',   301.00, 501.00],
    ['Saudi Arabia',501.00, 751.00],
    ['Bosnia & Herzegovina',251.00, 501.00],
    ['South Africa',501.00, 751.00],
    ['Panama',    501.00, 751.00],
    ['Congo DR',  501.00, 751.00],
    ['Uzbekistan',501.00, 751.00],
    ['Haiti',     751.00, 1001.00],
    ['Cabo Verde',751.00, 1001.00],
    ['Qatar',     751.00, 1001.00],
    ['Iraq',      751.00, 1001.00],
    ['New Zealand',1001.00, 1501.00],
    ['Jordan',    1001.00, 1501.00],
    ['Curacao',   1001.00, 1501.00],
]

with open(os.path.join(DATA_DIR, 'betting_odds.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['team','odds_low','odds_high','implied_prob_pct'])
    for t, lo, hi in odds_list:
        avg = (lo + hi) / 2
        imp = round(1/avg * 100, 1)
        w.writerow([t, lo, hi, imp])
print(f'[OK] 博彩赔率: {len(odds_list)} 支')

# ═══════════════════════════════════════
# 4. 最新动态
# ═══════════════════════════════════════
news = """
2026世界杯 — 赛前关键动态 (2026年6月11日)
=============================================

【重大伤病缺席】
- Jurrien Timber (荷兰) — 腹股沟伤，全程缺席
- Mohammed Kudus (加纳) — 股四头肌腱伤
- Wataru Endo (日本) — 脚伤，宣布退出国家队
- Rodrygo, Estevao, Eder Militao, Wesley (巴西) — 多人伤缺
- Billy Gilmour (苏格兰) — 膝伤
- Ezzalzouli (摩洛哥) — 右膝MCL损伤
- Nayef Aguerd (摩洛哥) — 恢复不完全

【出战存疑】
- Neymar (巴西) — 2级小腿撕裂，缺席首战
- Lamine Yamal (西班牙) — 左腿筋，首战待定
- Alphonso Davies (加拿大) — 腿筋，缺席首战
- Christian Pulisic (美国) — 小腿碰伤
- Bukayo Saka (英格兰) — 跟腱谨慎管理

【市场概况】
- 总身价最高: 法国 15.2亿EUR, 英格兰 13.1亿EUR, 西班牙 12.7亿EUR
- 夺冠赔率最低: 西班牙 5.50, 法国 5.70, 英格兰 7.50
- 首秀球队 (4队): Uzbekistan, Jordan, Cape Verde, Curacao
- 阿根廷 FIFA排名 #1 (卫冕冠军)
"""
with open(os.path.join(DATA_DIR, 'latest_news.txt'), 'w', encoding='utf-8') as f:
    f.write(news.strip())
print('[OK] 最新动态')

print('\n=== 数据采集完成 ===')
print(f'数据文件位于: {DATA_DIR}')
