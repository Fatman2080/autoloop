"""
strategy.py — 量化交易策略
改动：将 ATR 计算从 rolling(50) 改为 rolling(20)，调整 Keltner 倍数为 1.5x
理由：ATR(20) 对短期波动更敏感，1.5x 的 Keltner 通道更紧凑，
      能更及时捕捉趋势变化。历史结果显示 ATR 参数和通道宽度是有效的优化方向。
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
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
    
    # 改用 ATR(20) 替代 ATR(50)，对短期波动更敏感
    atr = tr.rolling(20).mean()
    vol_ma = volume.rolling(50).mean()

    # 调整 Keltner 通道宽度为 1.5x（原为 2.0x），使通道更紧凑
    keltner_upper = ema50 + 1.5 * atr
    keltner_lower = ema50 - 1.5 * atr

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

    # ── 做多系统（始终运行，25% 仓位） ──
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

    # ── 做空系统（仅在 EMA斜率熊市 + ADX强趋势 中激活，40% 仓位） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(36).max()

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