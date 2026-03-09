"""
改进策略：调整波动率自适应仓位的上下限

基于R211(val_score=3.2246)的改进：
1. R211策略在多空系统中都使用了波动率自适应仓位调整因子vol_mult
2. 当前vol_mult的范围为[0.5, 1.2]，计算方式为1.0/vol_ratio，其中vol_ratio=当前ATR/长期ATR均值
3. 当市场波动率较低时(vol_ratio<1)，1.0/vol_ratio>1，但被限制在1.2以内，限制了在低波动行情中的仓位提升潜力
4. 当市场波动率较高时(vol_ratio>2)，1.0/vol_ratio<0.5，但被限制在0.5以上，未能充分降低高风险环境下的仓位
5. 改进：将vol_mult的范围从[0.5, 1.2]调整为[0.3, 1.5]，允许在低波动时更大胆加仓(最大1.5倍)，在高波动时更谨慎减仓(最小0.3倍)
6. 预期：更好适应不同波动率环境，提高风险调整后收益
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

    # Keltner通道：做多使用3.0x，做空使用2.5x
    keltner_upper = ema50 + 3.0 * atr
    keltner_lower_tight = ema50 - 2.5 * atr  # 做空更严格

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 波动率自适应仓位（改进点：调整范围）──
    atr_pct = atr / close
    vol_regime = atr_pct.rolling(50).mean()
    vol_ratio = atr_pct / vol_regime
    # 改进：将范围从[0.5, 1.2]调整为[0.3, 1.5]
    vol_mult = np.clip(1.0 / vol_ratio, 0.3, 1.5).fillna(1.0)

    # ── 做多系统（30% 仓位 × 波动系数，ADX>25过滤）──
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
                long_signal.iloc[i] = 0.30 * vol_mult.iloc[i]
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.30 * vol_mult.iloc[i]

    # ── 做空系统（50% 仓位 × 波动系数，使用更严格的Keltner 2.5x）──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    atr_exit = atr14 * 2.5

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
                    and close.iloc[i] < keltner_lower_tight.iloc[i]  # 使用更严格的2.5x
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.50 * vol_mult.iloc[i]
        else:
            if close.iloc[i] > entry_price + atr_exit.iloc[i]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.50 * vol_mult.iloc[i]

    return (long_signal + short_signal).clip(-1.0, 1.0)