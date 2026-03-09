"""
strategy.py — 量化交易策略

基于R136（val_score=2.3791）的改进：
- 做空仓位从70%提升到80%（历史数据显示更高做空仓位往往val_score更高）
- 做空出场通道从36缩短到30（更紧密止损，保留更多利润）
- 保持波动率自适应仓位 + EMA150熊市检测 + ADX>25 + Keltner 2.5x + 成交量1.1x

改进理由：
- R75曾尝试L30%/S50%达到val_score=2.3323，说明高做空仓位有潜力
- R109/112/114等高val_score策略也显示高做空仓位有效性
- 更紧密的出场通道可以减少回撤
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
    vol_ma = volume.rolling(50).mean()

    keltner_upper = ema50 + 2.5 * atr
    keltner_lower = ema50 - 2.5 * atr

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

    # ── 波动率自适应仓位 ──
    atr_pct = atr / close
    vol_regime = atr_pct.rolling(50).mean()
    vol_ratio = atr_pct / vol_regime
    vol_mult = np.clip(1.0 / vol_ratio, 0.5, 1.2)
    vol_mult = vol_mult.fillna(1.0)

    # ── 做多系统（25% 仓位 × 波动系数） ──
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
                long_signal.iloc[i] = 0.25 * vol_mult.iloc[i]
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = 0.25 * vol_mult.iloc[i]

    # ── 做空系统（80% 仓位 × 波动系数，出场通道30） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(30).max()  # 从36缩短到30

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
                short_signal.iloc[i] = -0.80 * vol_mult.iloc[i]  # 从70%提升到80%
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -0.80 * vol_mult.iloc[i]

    return (long_signal + short_signal).clip(-1.0, 1.0)