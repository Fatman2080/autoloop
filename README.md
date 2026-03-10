# AutoLoop — AI 自主循环优化量化交易策略

让多个大模型并行竞争，自动迭代进化 BTC/USDT 交易策略。

## 它是什么

AutoLoop 是一个**全自动量化策略研究框架**：把策略优化的工作交给 LLM，让它们在一个封闭的沙盒中反复修改代码、回测、评估、保留或回滚——无需人工干预。

```
┌─────────────────────────────────────────────┐
│            每轮自动循环                       │
│                                             │
│  1. 并行调用多个 LLM（不同模型×不同温度）      │
│  2. 获取多个候选策略代码                      │
│  3. 逐个回测，选出最优                        │
│  4. 超越当前最佳 → 保留；否则 → 回滚          │
│  5. 记录结果，更新报告                        │
│  6. 回到第 1 步                              │
└─────────────────────────────────────────────┘
```

## 核心特点

- **多模型并行竞争**：同时调用 MiniMax、DeepSeek、GLM 等模型，每轮产生 N 个候选策略，优胜劣汰
- **全自动闭环**：LLM 生成策略 → 语法校验 → 回测评估 → Git 版本管理 → 自动决策保留/回滚
- **防作弊沙盒**：LLM 只能修改 `strategy.py`，不能碰回测引擎和评分逻辑
- **自动报告**：每次突破时自动生成 `REPORT.md`，包含完整指标和策略代码

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Fatman2080/autoloop.git
cd autoloop
```

### 2. 安装依赖

需要 [uv](https://docs.astral.sh/uv/) 包管理器：

```bash
uv venv && uv sync
```

### 3. 配置 API Key

复制环境变量模板并填入你的 API key（至少配一个）：

```bash
cp .env.example .env
```

```env
MINIMAX_API_KEY=your_key_here     # MiniMax (api.minimaxi.com)
DEEPSEEK_API_KEY=your_key_here    # DeepSeek (api.deepseek.com)
GLM_API_KEY=your_key_here         # 智谱 GLM (open.bigmodel.cn)
COINGLASS_API_KEY=your_key_here   # CoinGlass 衍生品数据（可选）
```

### 4. 准备数据

首次运行会自动从 Binance 下载 BTC/USDT 2 年 1h K 线数据并缓存到 `data/` 目录。

### 5. 启动自动循环

```bash
uv run python runner.py --max-rounds=200 --stale-limit=200
```

然后等待。每轮约 30-60 秒，策略会自动进化。

## 项目结构

```
├── strategy.py       # 策略代码（LLM 唯一可修改的文件）
├── backtest.py       # 回测入口
├── prepare.py        # 数据获取 + 回测引擎 + 评估函数（不可修改）
├── runner.py         # 自动循环控制器（多模型并行，需自行创建）
├── program.md        # LLM 的任务指令文档
├── results.tsv       # 每轮迭代记录
├── REPORT.md         # 最佳策略报告（自动生成）
├── pyproject.toml    # 项目依赖
├── .env.example      # API key 模板
└── data/             # K 线数据缓存（自动生成）
```

## 评分机制

```
score = Sharpe Ratio × (1 + 总收益率%)
```

- **Sharpe Ratio**：年化风险调整收益（越高越好）
- **收益加成**：收益越高分数越高，同时追求高 Sharpe 和高绝对收益
- 最大回撤 > 30% → 分数减半
- 交易次数 < 10 → 分数归零
- 策略代码 > 100 行 → 分数打 8 折

## 数据

| 类型 | 来源 | 列 |
|------|------|------|
| K 线 | Binance | open, high, low, close, volume |
| 资金费率 | CoinGlass | funding_rate |
| 未平仓合约 | CoinGlass | open_interest |
| 爆仓数据 | CoinGlass | liq_long_usd, liq_short_usd, liq_total_usd |
| 多空比 | CoinGlass | long_short_ratio |

数据按 70% / 15% / 15% 划分为训练集、验证集、测试集。LLM 只能看到训练集和验证集的回测结果。

## 策略规则

- 只能 `import pandas` 和 `import numpy`
- 不能使用未来数据（第 i 行信号只依赖第 0~i 行）
- 输出范围 -1.0 ~ 1.0（正=做多，负=做空）
- 函数签名不可改：`generate_signals(candles: pd.DataFrame) -> pd.Series`

## 添加新模型

在 `runner.py` 的 `PROVIDERS` 列表中添加配置，然后在 `.env` 中填入对应的 API key，即可自动参与并行竞争：

```python
{
    "name": "YourModel",
    "env_key": "YOUR_API_KEY",
    "url": "https://api.example.com/v1/chat/completions",
    "model": "model-name",
    "max_tokens": 8192,
}
```

## License

MIT
