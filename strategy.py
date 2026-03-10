"""
改进策略：做空仓位130% + 90%部分止盈

基于R289(val_score=4.0679)的改进：
1. 当前策略：做空仓位120% + 90%部分止盈
2. 改进：将做空仓位从120%提升到130%
3. 理由：
   - 历史规律：从55%到120%，每次+5%~10%仓位，val_score稳定提升（4.0679）
   - 当前回撤仅2.09%，风险完全可控
   - 130%仓位在极端做空行情时可获得更大收益
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

    # Keltner通道：做多使用3.0x，做空使用2.8x
    keltner_upper = ema50 + 3.0 * atr
    keltner_lower_tight = ema50 - 2.8 * atr

    # ── ADX 趋势强度指标 ──
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=candles.index).rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── 波动率自适应仓位（[0.25, 2.0]）──
    atr_pct = atr / close
    vol_regime = atr_pct.rolling(50).mean()
    vol_ratio = atr_pct / vol_regime
    vol_mult = np.clip(1.0 / vol_ratio, 0.25, 2.0).fillna(1.0)

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

    # ── 做空系统（130% 仓位 × 波动系数，90%部分止盈）──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    atr_exit = atr14 * 2.5
    atr_take_profit = atr14 * 2.5

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    entry_price = 0.0
    partial_closed = False

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower_tight.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_short = True
                entry_price = close.iloc[i]
                partial_closed = False
                short_signal.iloc[i] = -1.30 * vol_mult.iloc[i]  # 从1.20提高到1.30
        else:
            # 检查是否触发部分止盈（价格有利移动2.5x ATR）
            if not partial_closed and close.iloc[i] <= entry_price - atr_take_profit.iloc[i]:
                # 平掉90%仓位，剩余10%仓位
                short_signal.iloc[i] = -0.13 * vol_mult.iloc[i]  # 剩余10%仓位
                partial_closed = True
            elif close.iloc[i] > entry_price + atr_exit.iloc[i]:
                # 止损出场
                in_short = False
                partial_closed = False
            else:
                # 继续持有剩余仓位
                if partial_closed:
                    short_signal.iloc[i] = -0.13 * vol_mult.iloc[i]  # 剩余10%仓位
                else:
                    short_signal.iloc[i] = -1.30 * vol_mult.iloc[i]

    return (long_signal + short_signal).clip(-1.0, 1.0)