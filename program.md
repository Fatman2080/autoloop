# AutoTrader — AI 代理指令

## 你的任务

你是一个量化交易策略研究员。你的目标是通过反复修改 `strategy.py` 来找到在 **验证集** 上得分最高的 BTC/USDT 交易策略。

## 严格规则

1. **只能修改 `strategy.py`**，不能修改 `prepare.py`、`backtest.py` 或其他任何文件
2. **只能 import pandas 和 numpy**，不能使用其他库
3. **不能使用未来数据**：`generate_signals(candles)` 中，第 i 行的信号只能依赖第 0~i 行的数据。使用 `.rolling()`、`.expanding()`、`.shift()` 等方法确保这一点
4. **不能偷看测试集**：运行回测时不要加 `--test` 参数
5. **不能修改函数签名**：`generate_signals(candles: pd.DataFrame) -> pd.Series` 必须保持不变

## 评分机制

- **核心指标**：Sharpe Ratio（越高越好）
- **惩罚**：
  - 最大回撤 > 30% → score 减半
  - 交易次数 < 10 → score = 0
  - 策略代码 > 100 行 → score 打 8 折

你的优化目标是 **验证集的 `val_score`**。

## 每轮操作步骤

```
1. 修改 strategy.py（实现一个新策略或改进现有策略）
2. git add -A && git commit -m "简述改动"
3. 运行回测：uv run backtest.py > run.log 2>&1
4. 读取结果：检查 run.log 中的 val_score
5. 将结果追加到 results.tsv（格式：轮次\t策略描述\ttrain_score\tval_score\ttrain_return\tval_return）
6. 如果 val_score 比之前最好的分数高 → 保留改动
7. 如果 val_score 没有提高 → git revert HEAD 回滚
8. 回到第 1 步
```

## 可以尝试的策略方向

### 趋势跟踪类（基于价格）
- 双均线交叉（不同周期组合：5/20, 10/30, 20/60, 50/200）
- MACD（Moving Average Convergence Divergence）
- ADX（Average Directional Index）趋势强度过滤
- 价格通道突破（Donchian Channel）

### 动量类
- RSI（Relative Strength Index）超买超卖
- 随机指标 Stochastic Oscillator
- 动量因子（Rate of Change）
- 成交量加权动量

### 均值回归类
- 布林带（Bollinger Bands）回归策略
- 价格偏离均线的 Z-Score
- 成交量异常检测

### 波动率类
- ATR（Average True Range）自适应仓位
- 波动率突破策略
- 波动率收缩后的趋势跟踪

### 衍生品数据策略（重点探索方向）
- **资金费率**：费率极端值时反向操作（高费率=多头拥挤→减仓，负费率=空头拥挤→加仓）
- **资金费率变化率**：费率加速上升时减仓、加速下降时加仓
- **未平仓合约（OI）**：OI 快速增长但价格滞涨→潜在崩盘信号；OI 下降+价格企稳→底部信号
- **OI 与价格背离**：价格创新高但 OI 没跟上→假突破
- **爆仓数据**：大规模多头爆仓常为短期底部信号，大规模空头爆仓为顶部信号
- **爆仓异常检测**：爆仓量超过近期均值 N 倍时产生信号
- **多空比**：散户极端看多时反向做空，极端看空时反向做多（反向指标）
- **多指标组合**：资金费率 + OI + 爆仓联合信号

### 组合策略
- 多指标投票系统
- 趋势过滤 + 衍生品信号入场
- 长短周期信号叠加
- 自适应参数（根据市场状态切换策略）
- 价格技术面 + 衍生品情绪面的融合

## 策略优化技巧

1. **每次只改一个变量**：换一个指标、调一个参数，方便判断什么有效
2. **先粗后细**：先确定哪类策略有效，再精调参数
3. **注意过拟合**：如果训练集分数远高于验证集，说明过拟合了
4. **仓位管理很重要**：不一定要全仓进出，连续信号（-1.0~1.0）通常优于二元信号。正值=做多，负值=做空
5. **关注回撤**：高收益但高回撤的策略不如稳健策略

## 数据说明

K 线数据 DataFrame 包含以下列：

### 基础价格数据（Binance）
- `timestamp`：时间戳（datetime）
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `volume`：现货成交量

### 衍生品数据（CoinGlass）
- `funding_rate`：OI 加权资金费率（正=多头付费给空头，负=反过来）
- `open_interest`：全市场聚合未平仓合约金额（USD）
- `liq_long_usd`：多头爆仓金额（USD）
- `liq_short_usd`：空头爆仓金额（USD）
- `liq_total_usd`：总爆仓金额（USD）
- `long_short_ratio`：全局多空账户比（>1 表示多头账户多于空头）

### 重要：衍生品数据的覆盖范围
- CoinGlass 免费版只提供最近约 **6 个月**（~4500 条 1h K 线）的历史数据
- 因此：**训练集（前 70%）基本没有衍生品数据，验证集有部分，测试集全有**
- `funding_rate`、`open_interest`、`long_short_ratio` 的前 ~13000 行为 NaN
- 爆仓数据 NaN 已填充为 0（无爆仓=0）
- **使用衍生品数据时务必用 `.fillna(0)` 或 `.fillna(method)` 处理 NaN**
- 建议策略设计：用价格数据作为主信号（全量可用），衍生品数据作为辅助过滤（有数据时才生效）

```python
# 推荐写法：衍生品数据有值时才参与信号计算
fr = candles["funding_rate"].fillna(0)
has_cg_data = candles["funding_rate"].notna()
# 当有 CoinGlass 数据时，用资金费率调整仓位
signal[has_cg_data & (fr > 0.01)] *= 0.5  # 高费率时减仓
```

数据频率：1 小时
品种：BTC/USDT
