import pandas as pd
import numpy as np


def generate_signals(candles: pd.DataFrame) -> pd.Series:
    """
    改进方向：微调ATR分位数阈值和动量参数
    
    具体改动：
    1. ATR分位数阈值从70%提高到65%（比原80%更敏感，但比70%稍保守）
    2. ROC窗口从10改回12（避免过度灵敏）
    3. ROC动态阈值系数从0.4改回0.45（平衡灵敏度）
    4. 简化部分冗余代码
    
    理由分析：
    - round46达到val_score=31.6177是当前最高
    - 在此基础上进行参数微调，避免过度优化
    - ATR分位数65%在高波动时比70%稍保守，减少假信号
    - ROC窗口12比10更稳定
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    volume = candles["volume"]
    
    funding_rate = candles.get("funding_rate", pd.Series(0, index=candles.index)).fillna(0)
    
    # ATR计算
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    atr_pct = atr14 / close
    
    # 动态ATR阈值：滚动65%分位数（原70%）
    atr_threshold = atr_pct.rolling(window=100).quantile(0.65)
    atr_threshold = atr_threshold.fillna(0.025)
    
    # VWAP计算
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
    vwap_bias = (close - vwap) / vwap * 100
    price_above_vwap = close > vwap
    price_below_vwap = close < vwap
    
    # 布林带
    ema20 = close.ewm(span=20, adjust=False).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = ema20 + 2 * bb_std
    bb_lower = ema20 - 2 * bb_std
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
    bb_breakout_up = close > bb_upper
    bb_breakout_down = close < bb_lower
    bb_middle_zone = (bb_position > 0.4) & (bb_position < 0.6)
    
    # 长期趋势
    ema120 = close.ewm(span=120, adjust=False).mean()
    ema120_prev = ema120.shift(1)
    trend_up = ema120 > ema120_prev
    trend_down = ema120 < ema120_prev
    
    # 动量指标：ROC(12)与动态阈值（原10改为12，系数0.4改为0.45）
    roc = (close - close.shift(12)) / close.shift(12) * 100
    roc_std = roc.rolling(window=28).std()
    roc_dynamic_threshold = roc_std * 0.45
    momentum_strength = (roc.abs() - roc_dynamic_threshold) / roc_dynamic_threshold
    
    # 动态均线周期
    use_short = atr_pct > atr_threshold
    ema_fast = np.where(use_short, close.ewm(span=10, adjust=False).mean(),
                        close.ewm(span=20, adjust=False).mean())
    ema_slow = np.where(use_short, close.ewm(span=30, adjust=False).mean(),
                        close.ewm(span=60, adjust=False).mean())
    ema_fast = pd.Series(ema_fast, index=close.index)
    ema_slow = pd.Series(ema_slow, index=close.index)
    
    # 基础信号
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    volume_ma20 = volume.rolling(window=20).mean()
    signal_raw = pd.Series(0.0, index=candles.index)
    signal_raw[ema_fast > ema_slow] = 0.8
    signal_raw[ema_fast < ema_slow] = -0.8
    
    # 过滤条件
    gap_threshold = 0.008
    strong_trend = ema_diff_pct.abs() > gap_threshold
    volume_confirmed = volume > volume_ma20
    funding_threshold = 0.00003
    funding_bullish = funding_rate > funding_threshold
    funding_bearish = funding_rate < -funding_threshold
    strong_up_bias = vwap_bias > 1.0
    strong_down_bias = vwap_bias < -1.0
    
    # 动量确认
    momentum_confirmed_long = roc > roc_dynamic_threshold
    momentum_confirmed_short = roc < -roc_dynamic_threshold
    
    # 信号条件
    long_cond = ((signal_raw > 0) & strong_trend & volume_confirmed & 
                 price_above_vwap & momentum_confirmed_long)
    short_cond = ((signal_raw < 0) & strong_trend & volume_confirmed & 
                  price_below_vwap & momentum_confirmed_short)
    
    # 动态仓位计算
    long_base = np.full(len(close), 0.8)
    long_base = np.where(funding_bullish, long_base + 0.15, long_base)
    long_base = np.where(funding_bearish, long_base - 0.2, long_base)
    long_base = np.where(trend_up, np.minimum(long_base + 0.1, 1.0), long_base)
    long_base = np.where(trend_down, long_base - 0.15, long_base)
    long_base = np.where(strong_up_bias, np.minimum(long_base + 0.15, 1.0), long_base)
    long_base = np.where(bb_breakout_up, np.minimum(long_base + 0.2, 1.0), long_base)
    long_base = np.where(bb_middle_zone, long_base * 0.7, long_base)
    momentum_factor_long = np.clip(momentum_strength * 0.1, -0.2, 0.3)
    long_base = np.where(long_cond, np.clip(long_base + momentum_factor_long, 0, 1.0), long_base)
    
    short_base = np.full(len(close), -0.8)
    short_base = np.where(funding_bearish, short_base - 0.15, short_base)
    short_base = np.where(funding_bullish, short_base + 0.2, short_base)
    short_base = np.where(trend_down, np.maximum(short_base - 0.1, -1.0), short_base)
    short_base = np.where(trend_up, short_base + 0.15, short_base)
    short_base = np.where(strong_down_bias, np.maximum(short_base - 0.15, -1.0), short_base)
    short_base = np.where(bb_breakout_down, np.maximum(short_base - 0.2, -1.0), short_base)
    short_base = np.where(bb_middle_zone, short_base * 0.7, short_base)
    momentum_factor_short = np.clip(momentum_strength * 0.1, -0.3, 0.2)
    short_base = np.where(short_cond, np.clip(short_base - momentum_factor_short, -1.0, 0), short_base)
    
    # 合成最终信号
    final_signal = np.where(long_cond, long_base,
                np.where(short_cond, short_base, np.nan))
    
    final_signal = pd.Series(final_signal, index=candles.index)
    final_signal = final_signal.bfill().fillna(0)
    
    return final_signal.clip(-1.0, 1.0)