# 🏆 2026 FIFA World Cup Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/回测准确率-75%25-success)](src/backtest_june14.py)
[![Last Update](https://img.shields.io/badge/更新-2026.06.15-orange)]()

基于 **Dixon-Coles 泊松回归 + FIFA 排名 + 博彩赔率** 的 2026 世界杯比分预测系统。48 支球队、104 场比赛、每日更新。

---

## 🎯 它能做什么

- ⚽ **预测每场比赛的比分概率** — 不只告诉你谁赢，还告诉你 2-0 还是 1-1
- 📊 **12 个小组出线形势分析** — 实时积分推演
- 🏆 **蒙特卡洛 10,000 次模拟** — 夺冠概率、四强概率
- 📈 **回测验证** — 与实际赛果对比，持续校准

---

## 📸 预测样例

| 日期 | 比赛 | 预期比分 | 最常见 | 主胜 | 平 | 客胜 |
|------|------|----------|--------|------|-----|------|
| 6/16 | 🇫🇷 France vs Senegal | 2.2 - 1.0 | **2-0** | 66% | 20% | 15% |
| 6/16 | 🇦🇷 Argentina vs Algeria | 2.9 - 0.7 | **2-0** | 83% | 12% | 6% |
| 6/17 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England vs Ghana | 2.5 - 0.7 | **2-0** | 77% | 15% | 8% |
| 6/18 | 🇲🇽 Mexico vs South Korea | 1.7 - 1.2 | **1-1** | 50% | 24% | 26% |
| 6/19 | 🇺🇸 USA vs Australia | 1.7 - 1.2 | **1-1** | 50% | 24% | 26% |

> [查看完整 72 场小组赛预测 →](output/enhanced_group_stage_scores.md)

---

## 🧠 技术架构

```
数据采集 (Tavily/Bocha MCP)
        │
        ▼
┌──────────────────────────────┐
│  多模型集成预测引擎              │
│                              │
│  Dixon-Coles 泊松 (45%)      │  ← 预期进球 λ + 比分概率矩阵
│  FIFA 世界排名   (35%)       │  ← 2026/6/11 最新排名
│  博彩赔率       (15%)        │  ← BetMGM/FanDuel 市场
│  伤病调整       (5%)         │  ← 30+ 球员伤病追踪
└──────────────┬───────────────┘
               │
               ▼
    ┌──────────────────┐
    │  蒙特卡洛 10,000 次  │
    │  完整赛事模拟       │
    └──────────────────┘
               │
               ▼
      预测报告 + 置信度评估
```

### 核心模型

- **Dixon-Coles 泊松回归** — 每队独立攻击/防守参数，低比分相关性修正（0-0, 1-0, 0-1, 1-1）
- **Elo 评分** — K 值赛事分级 + 净胜球加成 + 回归均值
- **XGBoost** — 12 维特征集成学习
- **Stacking Ensemble** — 元模型融合 + Isotonic 概率校准

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.10
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
```

### 安装运行

```bash
git clone git@github.com:CARE-TH/worldcup-2026-predictor.git
cd worldcup-2026-predictor
pip install -r requirements.txt

# 运行增强版预测
python src/predict_enhanced.py

# 查看报告
cat output/enhanced_group_stage_scores.md
```

### 配合 Claude Code 使用

```bash
cd worldcup-2026-predictor
claude
# 说: "搜索最新赛果，更新预测模型"
```

---

## 📂 项目结构

```
worldcup-2026-predictor/
├── CLAUDE.md                        ← Claude Code 自动加载指令
├── README.md
├── requirements.txt                 ← 仅 3 个依赖
│
├── data/
│   ├── raw/                         ← 原始数据
│   │   ├── fixtures_2026.csv           赛程表
│   │   ├── match_history.csv           534 场历史比赛
│   │   └── team_attributes.csv         球队属性
│   └── processed/                   ← 模型输出
│       ├── team_params.csv             攻防参数
│       └── elo_ratings.csv             Elo 评分
│
├── models/                          ← 训练好的模型
│   └── xgb_model.pkl
│
├── src/                             ← 核心代码
│   ├── poisson_model.py               泊松引擎 ⭐
│   ├── predict_enhanced.py            增强预测 ⭐
│   ├── tournament_simulator.py        赛事模拟器
│   ├── elo_calculator.py             Elo 评分
│   ├── backtest_june14.py            回测脚本
│   └── ...
│
└── output/                          ← 预测报告
    ├── enhanced_group_stage_scores.md  增强版小组赛
    └── simulation_results.json         模拟结果
```

---

## 📊 预测精度

| 指标 | 结果 |
|------|------|
| 胜者预测 | **75%** (3/4) |
| 方向性判断 | 可靠 |
| 精确比分 | 不可预测（行业常态） |
| 总进球偏差 | 低估约 40%（持续校准中） |

> 6/14 回测详情: Germany 7-1 ✅ | Côte d'Ivoire 1-0 ✅ | Sweden 5-1 ✅ | Netherlands 2-2 ❌

---

## 🔄 每日更新流程

1. 搜索最新赛果 → 更新 `ACTUAL_RESULTS`
2. 调整状态参数（FORM_BOOST / INJURY）
3. 运行 `predict_enhanced.py`
4. 对比预测 vs 实际 → 校准模型
5. Git commit + push

---

## ⚠️ 免责声明

本预测仅供娱乐参考。足球比赛受天气、裁判、心理、运气等无数因素影响，任何模型都无法准确预知结果。**请勿用于赌博。**

---

## 📄 License

MIT © 2026
