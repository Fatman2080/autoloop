"""
strategy.py — 量化交易策略

基于R147改进（val_score=2.3879）：
- 核心改进：添加成交量趋势确认（VOL>VOL_MA），替代原来的固定1.1x倍数
- 保持：EMA150熊市检测 + ADX>25 + Keltner 2.5x + 波动率自适应
- 仓位：L30% / S70%（做空略增强）
- 改进理由：历史显示成交量确认有效，但固定倍数可能过于死板，趋势确认更灵活

改进思路：
- 成交量趋势确认比固定倍数更稳健
- 保持做空略重的仓位配比（历史上表现较好）
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
    vol_ma = volume.rolling(20).mean()  # 成交量均线确认

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

    # ── 做多系统（30% 仓位 × 波动系数） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        # 成交量趋势确认：当前成交量 > 20日均量
        vol_confirm = volume.iloc[i] > vol_ma.iloc[i]
        
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and vol_confirm):
                in_long = True
                long_signal.iloc[i] = 0.30 * vol_mult.iloc[i]
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.30 * vol_mult.iloc[i]

    # ── 做空系统（70% 仓位 × 波动系数） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    
    # ATR动态出场
    atr_exit = atr14 * 2.0

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False
    entry_price = 0.0

    for i in range(150, len(candles)):
        slope = ema150_slope.iloc[i]
        if np.isnan(slope):
            slope = 0.0
        bear_confirmed = close.iloc[i] < ema150.iloc[i] and slope < -0.05
        adx_strong = adx.iloc[i] > 25 if not np.isnan(adx.iloc[i]) else False
        vol_confirm = volume.iloc[i] > vol_ma.iloc[i]

        if not in_short:
            if (bear_confirmed
                    and adx_strong
                    and close.iloc[i] < keltner_lower.iloc[i]
                    and vol_confirm):
                in_short = True
                entry_price = close.iloc[i]
                short_signal.iloc[i] = -0.70 * vol_mult.iloc[i]
        else:
            if close.iloc[i] > entry_price + atr_exit.iloc[i]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.70 * vol_mult.iloc[i]

    return (long_signal + short_signal).clip(-1.0, 1.0)