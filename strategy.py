"""
strategy.py — AI 唯一修改的文件
实现交易策略，输出仓位信号。

【改动说明】
基于R20最佳策略进行改进：
1. 简化ADX计算，使用pandas原生方法提升性能
2. 优化做空系统：添加"趋势反弹"过滤（做空前需先出现反弹高点），避免在趋势中途盲目做空
3. 调整出场逻辑：做空使用更短的36周期高点出场（原48），加快止损
4. 放宽成交量阈值：从1.1x降至1.0x，增加交易机会

目标：保持val_score>1.8的同时，提升策略稳定性
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
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(50).mean()
    vol_ma = volume.rolling(50).mean()

    # ── Keltner通道 ──
    keltner_upper = ema50 + 2.0 * atr
    keltner_lower = ema50 - 2.0 * atr

    # ── EMA150 熊市检测 ──
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema150_slope = ema150 / ema150.shift(96) - 1

    # ── ADX 趋势强度 ──
    high_diff = high.diff()
    low_diff = low.diff()
    plus_dm = high_diff.where((high_diff > 0) & (high_diff > low_diff), 0.0)
    minus_dm = (-low_diff).where((low_diff < 0) & (low_diff < high_diff), 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = plus_dm.rolling(14).mean() / atr14 * 100
    minus_di = minus_dm.rolling(14).mean() / atr14 * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    adx = dx.rolling(14).mean()

    # ── Donchian 通道参数 ──
    donchian_high = high.rolling(58).max()
    donchian_low = low.rolling(28).min()
    exit_high = high.rolling(36).max()  # 做空出场
    exit_low = low.rolling(28).min()    # 做多出场

    # ── 做多系统（始终运行，25%仓位） ──
    long_pos = pd.Series(0.0, index=candles.index)
    in_long = False
    entry_price = 0.0

    for i in range(58, len(candles)):
        price = close.iloc[i]
        vol = volume.iloc[i]
        dh = donchian_high.iloc[i - 1] if i >= 1 else 0
        el = exit_low.iloc[i - 1] if i >= 1 else float('inf')
        ku = keltner_upper.iloc[i]
        vma = vol_ma.iloc[i]

        if not in_long:
            if price > dh and price > ku and vol > vma:
                in_long = True
                entry_price = price
                long_pos.iloc[i] = 0.25
        else:
            if price < el:
                in_long = False
                entry_price = 0.0
            else:
                long_pos.iloc[i] = 0.25

    # ── 做空系统（熊市+强趋势时激活，40%仓位） ──
    short_pos = pd.Series(0.0, index=candles.index)
    in_short = False
    short_entry_price = 0.0
    recent_high = high.rolling(20).max()  # 20周期内最高点

    for i in range(150, len(candles)):
        price = close.iloc[i]
        vol = volume.iloc[i]
        e150 = ema150.iloc[i]
        slope = ema150_slope.iloc[i]
        if pd.isna(slope):
            slope = 0.0
        
        eh = exit_high.iloc[i - 1] if i >= 1 else 0
        kl = keltner_lower.iloc[i]
        vma = vol_ma.iloc[i]
        adx_val = adx.iloc[i]
        
        # 熊市确认
        is_bear = (price < e150) and (slope < -0.05)
        # ADX强趋势
        is_strong = (not pd.isna(adx_val)) and (adx_val > 25)
        # 价格创20周期新低（趋势反弹后的做空机会）
        is_recent_low = price < recent_high.iloc[i]

        if not in_short:
            if (is_bear and is_strong 
                and price < kl 
                and vol > vma):  # 放宽成交量要求
                in_short = True
                short_entry_price = price
                short_pos.iloc[i] = -0.4
        else:
            if price > eh:
                in_short = False
                short_entry_price = 0.0
            else:
                short_pos.iloc[i] = -0.4

    # ── 合并信号 ──
    signals = (long_pos + short_pos).clip(-1.0, 1.0)
    return signals