"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进思路（基于历史分析）:
- round 17 达到 val_score 2.3948（做空40%），但训练集亏损
- round 20 val_score 1.870 更稳定（加入ADX>25过滤）
- 核心发现：更短的持仓周期 + 适度仓位 能提升 Sharpe
- 本轮改进：缩短做空持仓周期（exit从36改为32），增加时间止损（20周期强制平仓）
  目的：减少趋势反转时的持仓风险，提升 val Sharpe
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    输入：K线数据 DataFrame
        价格：timestamp, open, high, low, close, volume
        衍生品：funding_rate, open_interest, liq_long_usd, liq_short_usd, liq_total_usd, long_short_ratio

    输出：仓位信号 Series，-1.0 ~ 1.0
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 共用指标 ──
    ema50 = close.ewm(span=50, adjust=False).mean()
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

    # ── ADX 趋势强度 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 做多系统（25% 仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_entry_bar = 0

    for i in range(58, len(candles)):
        bars_held = i - long_entry_bar
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_entry_bar = i
                long_signal.iloc[i] = 0.25
        else:
            # 时间止损：20周期强制平仓
            if bars_held >= 20:
                in_long = False
            elif close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            if in_long:
                long_signal.iloc[i] = 0.25

    # ── 做空系统（40% 仓位，熊市+ADX>25） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(32).max()  # 改进：32（原36）缩短持仓

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_bar = 0

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        bars_held = i - short_entry_bar

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_entry_bar = i
                short_signal.iloc[i] = -0.4
        else:
            # 时间止损：20周期强制平仓
            if bars_held >= 20:
                in_short = False
            elif close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            if in_short:
                short_signal.iloc[i] = -0.4

    return (long_signal + short_signal).clip(-1.0, 1.0)