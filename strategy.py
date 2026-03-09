"""
strategy.py — 量化交易策略

改进自R138（val_score=2.0595）：
- 多空系统对称化：做多也改用ATR14动态出场（原固定28周期低点）
- ATR出场乘数优化：做空从2.0x改为2.5x，给予更大止盈空间
- 保持：EMA150熊市检测 + ADX>25 + Keltner 2.5x + 成交量1.1x + 波动率自适应
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
    atr = tr.rolling(50).mean()
    atr14 = tr.rolling(14).mean()
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.5 * atr
    keltner_lower = ema50 - 2.5 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 波动率自适应仓位 ──
    atr_pct = atr / close
    vol_regime = atr_pct.rolling(50).mean()
    vol_ratio = atr_pct / vol_regime
    vol_mult = np.clip(1.0 / vol_ratio, 0.5, 1.2).fillna(1.0)

    # ── 做多系统（30% 仓位 × 波动系数，ATR动态出场） ──
    entry_high = high.rolling(58).max()
    # ATR动态出场替代固定28周期低点
    long_atr_exit = atr14 * 2.5

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False
    long_entry_price = 0.0

    for i in range(58, len(candles)):
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_entry_price = close.iloc[i]
                long_signal.iloc[i] = 0.30 * vol_mult.iloc[i]
        else:
            # ATR动态出场：跌破入场价-2.5倍ATR时退出
            if close.iloc[i] < long_entry_price - long_atr_exit.iloc[i]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.30 * vol_mult.iloc[i]

    # ── 做空系统（70% 仓位 × 波动系数，ATR动态出场） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    # ATR动态出场，做空2.5xATR（原2.0x）
    short_atr_exit = atr14 * 2.5

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_price = 0.0

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
                short_entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.70 * vol_mult.iloc[i]
        else:
            # ATR动态出场：突破入场价+2.5倍ATR时退出
            if close.iloc[i] > short_entry_price + short_atr_exit.iloc[i]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.70 * vol_mult.iloc[i]

    return (long_signal + short_signal).clip(-1.0, 1.0)