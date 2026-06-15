# 2026世界杯预测系统 — 深度分析项目

## 项目定位
构建一个**可解释、可验证、可干预**的世界杯预测系统。不追求黑箱准确率，追求每一条预测结论都有清晰的推理链条和置信评估。

## 你的能力
- ✅ Tavily MCP（国际搜索）：查英文数据、赔率、新闻
- ✅ Bocha MCP（中文搜索）：查中文资讯、国内专家分析
- ✅ Python 编程：数据分析、建模、模拟
- ❌ 无实时联网 API（football-data.org 等外部 API 不可用，所有数据必须通过搜索或本地文件获取）

---

# 📋 第零步：初始化项目结构

开始工作前，先执行：

```bash
mkdir -p worldcup-prediction/{data/{raw,processed},models,src,output}
cd worldcup-prediction
```

然后将本文件（CLAUDE‑世界杯预测.md）的内容复制到项目根目录的 `CLAUDE.md`。之后你每次启动 `claude` 时会自动加载这份指令。

---

# 🔬 第一阶段：数据采集（完全通过搜索完成）

你的搜索策略是：**Tavily 搜国际数据，Bocha 搜中文数据，交叉验证。**

## 任务 1.1：获取2026世界杯赛程

```python
# 用 tavily 搜索
tavily search: "2026 FIFA World Cup full schedule group stage fixtures"
# 用 bocha 补充
bocha search: "2026世界杯 赛程表 48队 分组"
```

从搜索结果提取并存储为 `data/raw/fixtures_2026.csv`：
```
group, match_id, date, home_team, away_team, venue, stage
A, 1, 2026-06-11, Mexico, Canada, Mexico City, group
...
```

**赛程规则：** 2026世界杯48队 → 12个小组（每组4队） → 小组前2 + 8个成绩最好的第3名 → 32强淘汰赛

## 任务 1.2：获取各队历史比赛数据

```python
# 搜索主力联赛和世界杯历史数据
tavily search: "international football match results 2022-2026 CSV dataset"
tavily search: "FIFA World Cup qualifiers 2023-2025 results all teams"
tavily search: "Copa America 2024 Euro 2024 AFC Asian Cup 2023 results"
```

对每场比赛记录（存储到 `data/raw/match_history.csv`）：
```
date, home_team, away_team, home_goals, away_goals, tournament, neutral
2024-07-14, Spain, England, 2, 1, Euro 2024, Y
```

**覆盖范围：**
- 必须包含所有48支参赛队近4年的比赛
- 赛事优先级：世界杯 > 洲际杯 > 预选赛 > 友谊赛
- 如果某个队的比赛数据不足20场，标记为"数据不足"

## 任务 1.3：获取球队基本面数据

```python
# 搜索各队基本面
tavily search: "World Cup 2026 qualified teams FIFA ranking June 2026"
tavily search: "World Cup 2026 squads market value transfermarkt"
bocha search: "2026世界杯 32强 球队身价 排名"
```

对每支参赛队记录（存储到 `data/raw/team_attributes.csv`）：
```
team, fifa_rank, squad_value_billion, avg_age, world_cup_appearances, 
best_result, coach, star_player, playing_style
Brazil, 3, 1.2, 27.5, 22, Champion, Dorival Jr, Vinicius Jr, Attacking
```

## 任务 1.4：获取近期伤病和动态信息

```python
# 搜索关键动态
tavily search: "World Cup 2026 key injuries before tournament June 2026"
bocha search: "2026世界杯 伤病 大名单 最新"
tavily search: "World Cup 2026 odds favorites betting"
bocha search: "2026世界杯 夺冠赔率 最新"
```

提取关键信息存储到 `data/raw/latest_news.txt`

---

# 🧮 第二阶段：特征工程与统计建模

## 任务 2.1：数据清洗与时间加权

创建 `src/data_pipeline.py`，实现：

```python
# 权重规则
def calculate_weight(match_date, tournament, is_knockout):
    """
    核心思想：越近的比赛、越重要的赛事、权重越高
    """
    days_ago = (REFERENCE_DATE - match_date).days
    
    # 时间衰减：指数衰减，半衰期2年
    time_weight = 0.5 ** (days_ago / 730)
    
    # 赛事加成
    tournament_multiplier = {
        'World Cup': 3.0,
        'Continental Cup': 2.0,   # 美洲杯/欧洲杯/亚洲杯
        'Qualifier': 1.5,
        'Friendly': 0.8,
        'Confederations Cup': 1.5
    }
    
    # 淘汰赛加成
    knockout_bonus = 1.3 if is_knockout else 1.0
    
    return time_weight * tournament_multiplier[tournament] * knockout_bonus
```

**重要：** 必须做时间点检查——不能用2026年的数据去预测2018年的事情（回测时用）。

## 任务 2.2：泊松回归模型（核心引擎）

创建 `src/poisson_model.py`，实现双变量泊松回归：

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

def fit_attack_defense(matches, teams):
    """
    极大似然估计每队的 attack_strength 和 defense_strength
    
    模型：
    log(λ_home) = μ + attack_home + defense_away + home_advantage
    log(λ_away) = μ + attack_away + defense_home
    
    约束：∑attack = 0, ∑defense = 0
    """
    n = len(teams)
    # 参数向量：[mu, home_adv, attack[0..n-2], defense[0..n-2]]
    # 最后一个队的参数 = -sum(others) 以满足约束
    
    def negative_log_likelihood(params):
        mu = params[0]
        home_adv = params[1]
        a = np.append(params[2:2+n-1], -sum(params[2:2+n-1]))
        d = np.append(params[2+n-1:], -sum(params[2+n-1:]))
        
        ll = 0
        for _, match in matches.iterrows():
            h_idx = teams.index(match['home_team'])
            a_idx = teams.index(match['away_team'])
            
            lambda_h = np.exp(mu + a[h_idx] + d[a_idx] + home_adv * (1-match['neutral']))
            lambda_a = np.exp(mu + a[a_idx] + d[h_idx])
            
            ll += poisson.logpmf(match['home_goals'], lambda_h)
            ll += poisson.logpmf(match['away_goals'], lambda_a)
        
        return -ll
    
    # 初始值 + 约束优化
    result = minimize(negative_log_likelihood, 
                      x0=[0.5, 0.3] + [0]*(2*n-2),
                      method='L-BFGS-B')
    
    # 提取参数...
    return team_params_df
```

**输出** `data/processed/team_params.csv`：
```
team, attack_strength, defense_strength, overall_rating
Brazil, 0.85, -0.52, 1.37
...
```

## 任务 2.3：Elo 评分系统（基准线）

创建 `src/elo_calculator.py`：

```python
class EloRating:
    def __init__(self):
        self.ratings = {team: 1300 for team in all_teams}
        # 初始分：传统强队1500，中等1300，新军1100
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update(self, team_a, team_b, score_a, score_b, tournament):
        # K值：世界杯60，洲际杯40，预选赛30，友谊赛20
        k = {'World Cup': 60, 'Continental Cup': 40, 
             'Qualifier': 30, 'Friendly': 20}[tournament]
        
        # 净胜球加成：赢1球×1.0，2球×1.3，3球+×1.5
        goal_diff = abs(score_a - score_b)
        goal_bonus = {1: 1.0, 2: 1.3, 3: 1.5}.get(goal_diff, 1.75)
        
        # 实际结果（1/0.5/0）
        actual = 1 if score_a > score_b else (0.5 if score_a == score_b else 0)
        expected = self.expected_score(self.ratings[team_a], self.ratings[team_b])
        
        self.ratings[team_a] += k * goal_bonus * (actual - expected)
        self.ratings[team_b] += k * goal_bonus * ((1-actual) - (1-expected))
```

**输出** `data/processed/elo_ratings.csv`

## 任务 2.4：XGBoost 高级模型

创建 `src/xgboost_model.py`：

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# 特征构建
def build_features(match, team_params, elo_ratings, recent_forms):
    return {
        'attack_rating_diff': match.home.attack - match.away.attack,
        'defense_rating_diff': match.home.defense - match.away.defense,
        'elo_diff': elo_ratings[match.home] - elo_ratings[match.away],
        'form_diff_5': recent_form[match.home]['last_5_win_rate'] - 
                       recent_form[match.away]['last_5_win_rate'],
        'form_diff_10': recent_form[match.home]['last_10_win_rate'] - 
                        recent_form[match.away]['last_10_win_rate'],
        'rank_diff': team_rank[match.away] - team_rank[match.home],
        'value_diff_log': np.log1p(squad_value[match.home]) - 
                          np.log1p(squad_value[match.away]),
        'rest_days_diff': rest_days[match.home] - rest_days[match.away],
        'tournament_exp_diff': wc_appearances[match.home] - wc_appearances[match.away],
        'h2h_advantage': head_to_head[match.home][match.away],
        'is_group_stage': 1 if match.stage == 'group' else 0,
        'knockout_experience_diff': ko_experience[match.home] - ko_experience[match.away],
    }

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
# 按年份划分：2022训练/2023测试、2023训练/2024测试...

# 目标变量：1=主胜, 0=平局, -1=客胜
# 训练参数
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    early_stopping_rounds=30
)
```

**输出：** `models/xgb_model.json` + `data/processed/feature_importance.csv`

---

# 🎲 第三阶段：蒙特卡洛赛事模拟

## 任务 3.1：单场比赛预测引擎

创建 `src/match_predictor.py`：

```python
def predict_match(home_team, away_team, is_neutral, team_params, elo_ratings, xgb_model):
    """
    三模型加权预测，输出胜/平/负概率和最常见比分
    """
    # 1. 泊松模型：生成比分概率矩阵
    λ_h = exp(μ + attack_home + defense_away + home_adv * (1-is_neutral))
    λ_a = exp(μ + attack_away + defense_home)
    score_matrix = np.outer(poisson.pmf(range(0,8), λ_h), poisson.pmf(range(0,8), λ_a))
    poisson_h_win = np.sum(np.triu(score_matrix, 1))
    poisson_draw = np.sum(np.diag(score_matrix))
    poisson_a_win = np.sum(np.tril(score_matrix, -1))
    
    # 2. Elo 模型
    elo_h_win = 1 / (1 + 10**((elo_a - elo_h) / 400))
    elo_a_win = 1 / (1 + 10**((elo_h - elo_a) / 400))
    
    # 3. XGBoost 模型（如果用到了该场比赛的时间点）
    xgb_probs = xgb_model.predict_proba(features)[0]
    
    # 4. 加权融合（权重根据历史回测表现动态调整）
    weights = {'poisson': 0.35, 'elo': 0.20, 'xgb': 0.45}
    final_h_win = poisson_h_win*0.35 + elo_h_win*0.20 + xgb_probs[2]*0.45
    final_draw = poisson_draw*0.35 + (1-elo_h_win-elo_a_win)*0.20 + xgb_probs[1]*0.45
    final_a_win = poisson_a_win*0.35 + elo_a_win*0.20 + xgb_probs[0]*0.45
    
    return {
        'home_win_pct': round(final_h_win*100, 1),
        'draw_pct': round(final_draw*100, 1),
        'away_win_pct': round(final_a_win*100, 1),
        'most_likely_score': f"{most_common_home_goals}-{most_common_away_goals}",
        'confidence': 'high' if max(final_*) > 0.55 else 'medium' if max(final_*) > 0.4 else 'low',
        'model_agreement': 'unanimous' if all_agree else 'split'
    }
```

## 任务 3.2：完整赛事模拟器

创建 `src/tournament_simulator.py`：

```python
def simulate_tournament(n_simulations=10000, fixtures, team_params, elo, xgb):
    """
    蒙特卡洛模拟完整世界杯赛事
    
    流程：
    ┌─────────────────────────────────────────┐
    │  小组赛阶段（48队→32队）                   │
    │  for each group:                          │
    │    循环6场比赛                             │
    │    计算小组积分榜                          │
    │    小组前2 + 4个成绩最好的第3名 = 出线       │
    │    同分规则：净胜球→进球数→直接交锋          │
    └──────────┬──────────────────────────────┘
               ↓
    ┌─────────────────────────────────────────┐
    │  淘汰赛阶段（32队→1队）                   │
    │  单场决胜，平局→加时30分钟→点球            │
    │  加时模型：常规主客方差值缩小 40%           │
    │  点球模型：按历史罚点球胜率随机              │
    └─────────────────────────────────────────┘
    
    每次模拟记录：
    - 每场比赛的比分
    - 每队最终排名
    - 夺冠/四强/八强/小组出线
    """
    results = {team: {'champion': 0, 'final': 0, 'semi': 0, 
                       'quarter': 0, 'round_32': 0, 'group_stage': 0}
               for team in all_teams}
    
    for sim in range(n_simulations):
        # 模拟小组赛
        group_winners, group_seconds, best_thirds = simulate_group_stage()
        
        # 模拟淘汰赛
        champion = simulate_knockout_stage(
            group_winners + group_seconds + best_thirds
        )
        
        results[champion]['champion'] += 1
        # ... 更新其他轮次
        if sim % 1000 == 0:
            print(f"Progress: {sim}/{n_simulations}")
    
    # 转换为百分比
    for team in results:
        results[team] = {k: v/n_simulations*100 for k, v in results[team].items()}
    
    return results
```

**关键实现细节：**
- 同分时严格按 FIFA 规则判定：积分 → 净胜球 → 进球数 → 直接交锋 → 公平竞赛积分 → 抽签
- 淘汰赛加时阶段的进球率降为常规时间的 60%
- 点球决胜时，历史点球胜率数据作为先验概率

## 任务 3.3：出力度的校准评估

所有概率输出必须附带 **置信区间**：

| 置信度 | 判定标准 | 输出样式 |
|--------|---------|---------|
| 高 | 模型之间分歧 < 5% | ✅ 概率 ± 3% |
| 中 | 分歧 5-15% | ⚠️ 概率 ± 8% |
| 低 | 分歧 > 15% / 数据不足 | ❓ 概率 ± 15% |

---

# 🧠 第四阶段：深度推理分析

这一阶段是区别于普通统计预测的关键。**不要只输出数字，要输出完整的分析报告。**

## 任务 4.1：撰写12组小组出线形势分析

对每个小组输出一份 `output/group_X_analysis.md`，格式如下：

```markdown
# Group A: Mexico, Canada, [Team C], [Team D]

## 实力分层
- 🥇 出线大热门：Mexico（62%）
- 🥈 主要竞争者：Canada（48%）
- 🥉 黑马潜力：[Team C]（22%）
- 💀 陪跑：[Team D]（8%）

## 核心对决
**6月15日 Mexico vs Canada** — 决定小组头名
泊松预测：Mexico 1.8 - 1.2 Canada | Mexico胜率41% 平局28%
推演：如果Mexico赢，基本锁定头名；平局则悬念留到最后

## 关键变量
1. Mexico主场优势（主办国）：+8%胜率加成
2. Canada近期状态火热（近10场7胜）：但世界杯经验不足
3. [Team C]防守反击风格：可能爆冷逼平强队

## 最可能积分榜
1. Mexico 7分 ✓
2. Canada 4分 ✓
3. [Team C] 3分（可能以最好第3名出线）
4. [Team D] 1分 ✗
```

## 任务 4.2：撰写Top 10关键战役深度分析

如果搜索到最新的伤病/阵容信息，务必纳入分析框架：

```
# 关键战役：Brazil vs Argentina（1/8决赛或更晚）

## 数据画像
Brazil: attack=+0.85, defense=-0.52, Elo=1580
Argentina: attack=+0.71, defense=-0.48, Elo=1540

## 胜负概率（模型输出）
Brazil 42% | Draw 28% | Argentina 30%

## 战术推演
Brazil的边路爆破（Vinicius + Raphinha） vs Argentina的边后卫短板
→ 但Argentina的中路控制（De Paul + Enzo）可以切断锋线供给
→ 关键：谁的比赛节奏？快节奏→Brazil，慢节奏→Argentina

## 最新变量（如有搜索到的最新信息）
- [如果搜到] Neymar已恢复训练并进入大名单 → Brazil胜率+5%
- [如果搜到] Messi最后一届世界杯的心理加成
- [如果搜到] 天气/场地条件

## 置信度评估
⚠️ 中等（模型分歧11%，且受核心球员状态影响大）
```

## 任务 4.3：综合预测报告

最后生成 `output/final_report.md`，格式：

```markdown
# 2026世界杯预测报告

## 夺冠概率 Top 10
| 排名 | 球队 | 夺冠概率 | 四强概率 | 小组出线概率 | 赛区 |
|------|------|---------|---------|------------|------|
| 1 | Brazil | 14.2% | 38% | 92% | South America |
| 2 | France | 12.8% | 35% | 88% | Europe |
| 3 | Argentina | 11.5% | 32% | 85% | South America |
| ... | ... | ... | ... | ... | ... |

## 黑马榜（夺冠概率 < 3% 但四强概率 > 10%）
- Uruguay：淘汰赛经验丰富，抽签有利
- Japan：连续三届小组出线，风格克制欧洲队

## 冷门预警
- 最有可能的小组赛翻车：...（哪支种子队可能小组不出线）
- 最有可能的1/8决赛冷门：...（排名差大但模型不看好强队）

## 模型可靠性自评
- 2018年回测准确率：小组赛51%，淘汰赛62%
- 2022年回测准确率：小组赛53%，淘汰赛58%
- 2026年预测置信度：总体中高，但淘汰赛路径依赖强——个别抽签/红牌等随机事件会大幅改变结果
```

---

# ✅ 第五阶段：回测验证（不可跳过）

## 任务 5.1：用2018和2022世界杯验证

创建 `src/backtest.py`：

```python
# 流程
for world_cup_year in [2018, 2022]:
    # 只使用该届世界杯之前的比赛数据训练模型
    train_data = matches[matches['date'] < f'{world_cup_year}-01-01']
    
    # 用训练好的模型预测该届世界杯每场比赛
    predictions = predict_each_match(world_cup_year_fixtures, model)
    
    # 比较预测 vs 实际
    accuracy = calculate_accuracy(predictions, actual_results)
    log_loss = calculate_log_loss(predictions, actual_results)
    brier = calculate_brier_score(predictions, actual_results)
    
    print(f"{world_cup_year}:")
    print(f"  - 比赛预测准确率: {accuracy['match']:.1f}%")
    print(f"  - 小组出线预测准确率: {accuracy['qualify']:.1f}%")
    print(f"  - 冠军预测: 模型={'Brazil' if ... else ...} vs 实际={'France' if ... else ...}")
```

**注意：** 回测不是为了得到一个好看的数字，而是为了理解模型的偏差。如果模型系统性高估/低估某类球队（比如总是高估南美队），要在最终报告中说明。

---

# 📊 输出文件清单

项目完成后应包含以下文件：

```
worldcup-prediction/
├── CLAUDE.md                          ← 本文档
├── data/
│   ├── raw/
│   │   ├── fixtures_2026.csv          ← 赛程（搜出来的）
│   │   ├── match_history.csv          ← 历史比赛（搜出来的）
│   │   ├── team_attributes.csv        ← 球队属性（搜出来的）
│   │   └── latest_news.txt            ← 最新动态（搜出来的）
│   └── processed/
│       ├── team_params.csv            ← 泊松模型输出
│       ├── elo_ratings.csv            ← Elo评分
│       └── feature_importance.csv     ← XGBoost特征重要性
├── models/
│   └── xgb_model.json                 ← XGBoost模型文件
├── src/
│   ├── data_pipeline.py              ← 数据清洗
│   ├── poisson_model.py              ← 泊松回归
│   ├── elo_calculator.py             ← Elo评分
│   ├── xgboost_model.py              ← XGBoost
│   ├── match_predictor.py            ← 单场预测引擎
│   ├── tournament_simulator.py       ← 赛事模拟器
│   └── backtest.py                   ← 回测验证
└── output/
    ├── group_A_analysis.md            ← 小组分析（12份）
    ├── group_B_analysis.md
    ├── ...
    ├── key_matches_analysis.md        ← 关键战役分析（Top 10）
    └── final_report.md                ← 最终预测报告
```

---

# ⚠️ 重要约束

1. **数据完整性优先于数据量**：确保48支参赛队的数据完整，不要为了追求数据量大而混入低质量数据
2. **时间点一致性**：在做回测时，绝对不能用未来的数据预测过去——必须模拟当时的信息环境
3. **不确定性须量化**：每个概率输出都附带置信度（高/中/低），避免给出虚假精确的数值
4. **分歧是财富**：当泊松、Elo、XGBoost三个模型给出分歧结果时，不要强行统一——把分歧写入报告，说明"为什么不同模型看法不同"
5. **每完成一步要我确认**：完成一个阶段后，给我看成果，确认后再进入下一阶段

---

# ▶️ 启动指令

```bash
# 第一步：创建项目目录
mkdir -p /mnt/c/Users/你的用户名/worldcup-prediction/{data/{raw,processed},models,src,output}

# 第二步：进入项目目录
cd /mnt/c/Users/你的用户名/worldcup-prediction

# 第三步：把本文件保存为 CLAUDE.md
#（复制本文件内容粘贴进去）

# 第四步：启动Claude Code
claude
```

Claude Code 启动后会自动读入这份 CLAUDE.md，然后开始执行第一阶段。
完成一个阶段后，停下来问我的意见，再进入下一阶段。
