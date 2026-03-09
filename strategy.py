"""
strategy.py — 量化交易策略
改进自R20最佳策略：调整仓位配比为 L30%/S50%
"""

import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    输入：K线数据 DataFrame
        价格数据：timestamp, open, high, low, close, volume
        衍生品数据：funding_rate, open_interest, liq_long_usd, liq_short_usd, 
                   liq_total_usd, long_short_ratio

    输出：仓位信号 Series，值在 -1.0 ~ 1.0 之间
    
    策略：独立叠加多空系统 + EMA150斜率熊市检测 + ADX趋势强度
    - 做多系统（始终运行）：Donchian(58h) + Keltner上轨(2.0x) + 成交量 → 30%
    - 做空系统（仅熊市+强趋势）：Keltner下轨(2.0x) + 成交量 + 熊市确认 + ADX>25 → 50%
    - 仓位调整：做多25%→30%，做空40%→50%，更充分捕捉熊市机会
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

    # ── 做多系统（始终运行，30% 仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(30).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = 0.30
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.30

    # ── 做空系统（仅在 EMA斜率熊市 + ADX强趋势 中激活，50% 仓位） ──
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
                short_signal.iloc[i] = -0.50
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.50

    return (long_signal + short_signal).clip(-1.0, 1.0)