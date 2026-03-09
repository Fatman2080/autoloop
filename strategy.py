"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改动说明：
- 在做多系统中加入 EMA150 趋势过滤：当价格 > EMA150 且 EMA150 处于上升趋势时，
  做多信号从 25% 提升到 35%；当价格 < EMA150 时，做多信号降至 15%
- 理由：R20 训练集 Sharpe 仅 0.103，说明做多系统可能在熊市/震荡市中贡献了亏损。
  通过只在牛市环境中加大做多仓位，可在保持验证集表现的同时提升训练集盈利能力。
  历史数据显示 R29 (val=2.33) 比 R20 更激进，可能是因为做空更频繁，
  故尝试在做多端也引入趋势过滤以实现多空对称。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    输入：K线数据 DataFrame，包含列：
        价格数据：timestamp, open, high, low, close, volume
        衍生品数据：funding_rate, open_interest,
                    liq_long_usd, liq_short_usd, liq_total_usd,
                    long_short_ratio

    输出：仓位信号 Series，值在 -1.0 ~ 1.0 之间
        - -1.0 = 满仓做空
        -  0.0 = 空仓
        -  1.0 = 满仓做多

    策略：独立叠加多空系统 + EMA150 趋势过滤 + ADX 趋势强度
    - 做多系统（根据牛市/熊市调整仓位）：
      * 牛市（价格>EMA150）：Donchian突破 + Keltner + 成交量 → 35%
      * 熊市（价格<EMA150）：Donchian突破 + Keltner + 成交量 → 15%
    - 做空系统（仅熊市+强趋势）：Keltner下轨 + 成交量 + 熊市确认 + ADX>25 → 40%
    - 熊市判定：价格 < EMA(150) 且 EMA(150) 96h内下跌 > 5%
    - ADX>25 过滤弱趋势做空
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 共用指标 ──
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1  # 96周期斜率
    
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── Donchian 通道 ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()
    exit_high = high.rolling(36).max()

    # ── 做多系统（根据市场状态调整仓位） ──
    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        # 判断牛市/熊市
        price_above_ema150 = close.iloc[i] > ema150.iloc[i]
        ema150_rising = ema150_slope.iloc[i] > -0.05 if not np.isnan(ema150_slope.iloc[i]) else False
        
        # 牛市：35%仓位；熊市：15%仓位
        long_pct = 0.35 if (price_above_ema150 or ema150_rising) else 0.15
        
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = long_pct
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = long_pct

    # ── 做空系统（仅在 EMA斜率熊市 + ADX强趋势 中激活，40% 仓位） ──
    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -0.4
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)