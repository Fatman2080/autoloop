"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

改进说明 (R21):
- 目标: 在低波动市场增加仓位，高波动市场减少仓位
- 方法: 使用 ATR/close 作为波动率指标，动态调整仓位乘数
- 保持原策略(R20)的开仓/出场逻辑不变，只调整仓位权重
- 预期: 在震荡/低波动期获得更多收益，同时在高波动期控制回撤

原策略: 独立叠加多空系统 + EMA斜率熊市检测 + ADX趋势强度
- 做多: 25% * vol_multiplier
- 做空: 40% * vol_multiplier (仅熊市+ADX>25)
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

    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── ATR波动率仓位调整因子 ──
    # 计算ATR占价格的百分比，作为波动率指标
    atr_pct = atr / close
    # 波动率越低，仓位乘数越高
    # 基准: atr_pct ≈ 0.02 (2%) 时，乘数为 1.0
    # 范围: 0.5x ~ 2.0x
    vol_multiplier = (0.02 / atr_pct.clip(0.005, 0.1)).clip(0.5, 2.0)
    # 平滑处理
    vol_multiplier = vol_multiplier.rolling(5).mean()

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

    # ── 做多系统（始终运行，动态仓位） ──
    entry_high = high.rolling(58).max()
    exit_low = low.rolling(28).min()

    long_signal = pd.Series(0.0, index=candles.index)
    in_long = False

    for i in range(58, len(candles)):
        # 动态仓位 = 基础25% * 波动率调整因子
        pos_size = 0.25 * vol_multiplier.iloc[i]
        
        if not in_long:
            if (close.iloc[i] > entry_high.iloc[i - 1]
                    and close.iloc[i] > keltner_upper.iloc[i]
                    and volume.iloc[i] > 1.1 * vol_ma.iloc[i]):
                in_long = True
                long_signal.iloc[i] = pos_size
        else:
            if close.iloc[i] < exit_low.iloc[i - 1]:
                in_long = False
            else:
                long_signal.iloc[i] = pos_size

    # ── 做空系统（仅在 EMA斜率熊市 + ADX强趋势 中激活，动态仓位） ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1
    exit_high = high.rolling(36).max()

    short_signal = pd.Series(0.0, index=candles.index)
    in_short = False

    for i in range(150, len(candles)):
        # 动态仓位 = 基础40% * 波动率调整因子
        pos_size = 0.40 * vol_multiplier.iloc[i]
        
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
                short_signal.iloc[i] = -pos_size
        else:
            if close.iloc[i] > exit_high.iloc[i - 1]:
                in_short = False
            else:
                short_signal.iloc[i] = -pos_size

    return (long_signal + short_signal).clip(-1.0, 1.0)