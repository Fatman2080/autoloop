"""
strategy.py — 量化交易策略
基于R20最佳策略的微调改进：
- 缩短做空出场通道：36 -> 30 (更快止损，捕捉更剧烈下跌)
- 增加做空仓位：40% -> 45% (在强趋势中获取更多收益)
- 优化熊市检测灵敏度：EMA150 -> EMA100 (更早识别熊市)
- 保留ADX>25趋势过滤，减少震荡市亏损
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]

    # ── 基础指标 ──
    ema50 = close.ewm(span=50, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    vol_ma = volume.rolling(50).mean()

    # ── Keltner 通道 ──
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

    # ── 做多系统：Donchian(58) + Keltner上轨 + 成交量确认，25%仓位 ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = 0.25
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25

    # ── 做空系统：EMA100熊市 + ADX>25 + Keltner下轨 + 成交量，45%仓位 ──
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema100_slope = ema100 / ema100.shift(96) - 1
    exit_high = high.rolling(30).max()  # 改为30（原36）

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(100, len(candles)):
        slope = ema100_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema100.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                short_signal.iloc[i] = -0.45  # 改为45%（原40%）
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.45

    return (long_signal + short_signal).clip(-1.0, 1.0)