"""
改进策略：降低做空仓位+优化ATR出场

基于R194(val_score=3.0655)改进：
1. 做空仓位从60%降低到50%：减少极端行情回撤，提高夏普比率稳定性
2. ATR出场倍数从2.8x微调到2.5x：更早获利了结，提升资金效率
3. 保持做多35%仓位和ADX>25阈值不变
4. 理由：验证集显示趋势行情，但做空系统仓位略重，适当降低仓位可以
   在保持收益的同时降低回撤，从而提升Sharpe Ratio
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

    # ── 做多系统（35% 仓位 × 波动系数，ADX>25过滤）──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(30).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        adx_value = adx.iloc[i] if not np.isnan(adx.iloc[i]) else 0
        adx_ok = adx_value > 25
        
        if not in_long:
            if (adx_ok
                    and close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = 0.35 * vol_mult.iloc[i]
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.35 * vol_mult.iloc[i]

    # ── 做空系统（50% 仓位 × 波动系数，ATR动态出场2.5x）──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    atr_exit = atr14 * 2.5  # 从2.8x降低到2.5x

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    entry_price = 0.0

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
                entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.50 * vol_mult.iloc[i]  # 从-0.60降低到-0.50
        else:
            if close.iloc[i] > entry_price + atr_exit.iloc[i]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.50 * vol_mult.iloc[i]

    return (long_signal + short_signal).clip(-1.0, 1.0)