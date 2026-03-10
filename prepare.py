"""
prepare.py — 固定不动的基础设施
负责：数据获取（Binance K线 + CoinGlass 衍生品数据）、回测引擎、评估函数
"""

import os
import time
import math
import requests
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. 数据获取
# ─────────────────────────────────────────────

SYMBOL = "BTCUSDT"
SYMBOL_CG = "BTC"  # CoinGlass 用不带 USDT 的符号
INTERVAL = "1h"
YEARS = 2

CG_BASE = "https://open-api-v4.coinglass.com"


def _load_coinglass_key() -> str:
    """从 .env 文件读取 CoinGlass API key"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("COINGLASS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("COINGLASS_API_KEY", "")


# ── Binance K 线 ──

def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """从 Binance 公开 API 分页拉取 K 线"""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_klines.extend(batch)
        current = batch[-1][0] + 1
        time.sleep(0.1)

    return all_klines


def _klines_to_dataframe(raw: list) -> pd.DataFrame:
    """将 Binance 原始 K 线转为干净的 DataFrame"""
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp")
    return df


# ── CoinGlass 衍生品数据 ──

def _cg_paginated_fetch(
    endpoint: str, params: dict, api_key: str,
    start_ms: int, end_ms: int, label: str,
) -> list:
    """
    CoinGlass API 分页拉取（每页最多 1000 条）。
    自动根据返回数据的最后时间戳翻页，带退避重试。
    """
    headers = {"CG-API-KEY": api_key}
    all_data = []
    current_start = start_ms
    retries = 0
    max_retries = 5

    while current_start < end_ms:
        p = {**params, "start_time": current_start, "end_time": end_ms, "limit": 1000}

        try:
            resp = requests.get(
                f"{CG_BASE}{endpoint}", params=p, headers=headers, timeout=30
            )
        except requests.RequestException as e:
            print(f"  [CoinGlass] {label} 网络错误: {e}", flush=True)
            retries += 1
            if retries > max_retries:
                break
            time.sleep(5 * retries)
            continue

        # HTTP 429 或 JSON body 中的 429
        if resp.status_code == 429:
            retries += 1
            wait = min(10 * retries, 60)
            print(f"  [CoinGlass] HTTP 429，等待 {wait}s（重试 {retries}/{max_retries}）", flush=True)
            if retries > max_retries:
                break
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"  [CoinGlass] {label} HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
            break

        body = resp.json()
        code = str(body.get("code", ""))

        if code == "429":
            retries += 1
            wait = min(10 * retries, 60)
            print(f"  [CoinGlass] API 429，等待 {wait}s（重试 {retries}/{max_retries}）", flush=True)
            if retries > max_retries:
                break
            time.sleep(wait)
            continue

        if code != "0" or not body.get("data"):
            print(f"  [CoinGlass] {label} 出错: code={code} msg={body.get('msg', '')}", flush=True)
            break

        retries = 0  # 成功后重置重试计数
        batch = body["data"]
        all_data.extend(batch)
        page = len(all_data) // 1000 + 1
        print(f"    {label}: 已拉取 {len(all_data)} 条（第 {page} 页）", flush=True)

        last_time = batch[-1].get("time") or batch[-1].get("t")
        if last_time is None or last_time >= end_ms or len(batch) < 1000:
            break
        current_start = last_time + 1
        time.sleep(1.5)  # 每页间隔 1.5 秒，避免触发限流

    return all_data


def _fetch_funding_rate(api_key: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """获取 OI 加权资金费率（1h OHLC）"""
    print("  [CoinGlass] 拉取资金费率...")
    data = _cg_paginated_fetch(
        endpoint="/api/futures/funding-rate/oi-weight-history",
        params={"symbol": SYMBOL_CG, "interval": INTERVAL},
        api_key=api_key,
        start_ms=start_ms, end_ms=end_ms,
        label="funding_rate",
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df["funding_rate"] = df["close"].astype(float)
    return df[["timestamp", "funding_rate"]]


def _fetch_open_interest(api_key: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """获取聚合未平仓合约（1h OHLC）"""
    print("  [CoinGlass] 拉取未平仓合约...")
    data = _cg_paginated_fetch(
        endpoint="/api/futures/open-interest/aggregated-history",
        params={"symbol": SYMBOL_CG, "interval": INTERVAL},
        api_key=api_key,
        start_ms=start_ms, end_ms=end_ms,
        label="open_interest",
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df["open_interest"] = df["close"].astype(float)
    return df[["timestamp", "open_interest"]]


def _fetch_liquidations(api_key: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """获取聚合爆仓数据（1h）"""
    print("  [CoinGlass] 拉取爆仓数据...")
    data = _cg_paginated_fetch(
        endpoint="/api/futures/liquidation/aggregated-history",
        params={
            "exchange_list": "Binance,OKX,Bybit,Bitget,dYdX",
            "symbol": SYMBOL_CG,
            "interval": INTERVAL,
        },
        api_key=api_key,
        start_ms=start_ms, end_ms=end_ms,
        label="liquidations",
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["aggregated_long_liquidation_usd", "aggregated_short_liquidation_usd"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            df[col] = 0.0
    df["liq_long_usd"] = df["aggregated_long_liquidation_usd"]
    df["liq_short_usd"] = df["aggregated_short_liquidation_usd"]
    df["liq_total_usd"] = df["liq_long_usd"] + df["liq_short_usd"]
    return df[["timestamp", "liq_long_usd", "liq_short_usd", "liq_total_usd"]]


def _fetch_long_short_ratio(api_key: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """获取全局多空账户比"""
    print("  [CoinGlass] 拉取多空比...")
    data = _cg_paginated_fetch(
        endpoint="/api/futures/global-long-short-account-ratio/history",
        params={
            "exchange": "Binance",
            "symbol": "BTCUSDT",
            "interval": INTERVAL,
        },
        api_key=api_key,
        start_ms=start_ms, end_ms=end_ms,
        label="long_short_ratio",
    )
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df["long_short_ratio"] = df["global_account_long_short_ratio"].astype(float)

    return df[["timestamp", "long_short_ratio"]]


# ── 合并下载 ──

def download_data(force: bool = False) -> pd.DataFrame:
    """
    下载 BTC/USDT 历史数据（K线 + 衍生品指标），带本地缓存。

    返回 DataFrame 包含列：
        timestamp, open, high, low, close, volume,
        funding_rate, open_interest,
        liq_long_usd, liq_short_usd, liq_total_usd,
        long_short_ratio
    """
    cache_file = DATA_DIR / f"{SYMBOL}_{INTERVAL}_{YEARS}y_enriched.csv"

    if cache_file.exists() and not force:
        print(f"[数据] 使用本地缓存: {cache_file.name}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        return df

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - YEARS * 365 * 24 * 3600 * 1000

    # 1) Binance K 线
    print(f"[数据] 从 Binance 下载 {SYMBOL} {INTERVAL} K线（近{YEARS}年）...")
    raw = _fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)
    df = _klines_to_dataframe(raw)
    print(f"  K 线: {len(df)} 条")

    # 2) CoinGlass 衍生品数据
    api_key = _load_coinglass_key()
    if api_key:
        print(f"[数据] 从 CoinGlass 下载衍生品数据...")

        fr_df = _fetch_funding_rate(api_key, start_ms, end_ms)
        time.sleep(5)
        oi_df = _fetch_open_interest(api_key, start_ms, end_ms)
        time.sleep(5)
        liq_df = _fetch_liquidations(api_key, start_ms, end_ms)
        time.sleep(5)
        ls_df = _fetch_long_short_ratio(api_key, start_ms, end_ms)

        for extra_df, name in [
            (fr_df, "funding_rate"),
            (oi_df, "open_interest"),
            (liq_df, "liquidations"),
            (ls_df, "long_short_ratio"),
        ]:
            if not extra_df.empty:
                extra_df["timestamp"] = extra_df["timestamp"].dt.floor("h")
                extra_df = extra_df.drop_duplicates(subset="timestamp", keep="last")
                df = df.merge(extra_df, on="timestamp", how="left")
                matched = df[extra_df.columns[1]].notna().sum()
                print(f"  {name}: {len(extra_df)} 条，匹配 {matched} 条")
            else:
                print(f"  {name}: 无数据（跳过）")
    else:
        print("[数据] 未找到 CoinGlass API key，跳过衍生品数据")

    # 确保所有列都存在（即使 CoinGlass 没数据）
    for col in ["funding_rate", "open_interest",
                "liq_long_usd", "liq_short_usd", "liq_total_usd",
                "long_short_ratio"]:
        if col not in df.columns:
            df[col] = np.nan

    # 前向填充衍生品数据中的缺失值（这些指标变化缓慢，前向填充合理）
    fill_cols = ["funding_rate", "open_interest", "long_short_ratio"]
    df[fill_cols] = df[fill_cols].ffill()

    # 爆仓数据缺失填 0（没有爆仓就是 0）
    for col in ["liq_long_usd", "liq_short_usd", "liq_total_usd"]:
        df[col] = df[col].fillna(0.0)

    df.to_csv(cache_file, index=False)
    print(f"[数据] 已缓存到: {cache_file.name}（{len(df)} 行 x {len(df.columns)} 列）")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    切分数据为训练集 / 验证集 / 测试集（70% / 15% / 15%）
    """
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy().reset_index(drop=True)
    val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test = df.iloc[val_end:].copy().reset_index(drop=True)

    print(f"[数据] 切分完成: 训练={len(train)} 验证={len(val)} 测试={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────
# 2. 回测引擎
# ─────────────────────────────────────────────

FEE_RATE = 0.001      # 0.1% 手续费
SLIPPAGE_RATE = 0.0005 # 0.05% 滑点


def backtest(
    signal: pd.Series,
    candles: pd.DataFrame,
    initial_capital: float = 10000.0,
) -> dict:
    """
    根据仓位信号进行回测（支持多空双向）。

    参数:
        signal: 仓位比例序列，值在 [-1.0, 1.0] 之间
                正值 = 做多（买入 BTC），负值 = 做空（卖空 BTC）
        candles: K 线数据，必须包含 open, high, low, close 列
        initial_capital: 初始资金（USDT）

    返回:
        dict，包含 equity_curve, trades, final_capital 等

    规则:
        - 信号在当前 K 线收盘时产生，下一根 K 线开盘价执行
        - 手续费 0.1%，滑点 0.05%
        - 仓位范围 -100% ~ +100%，不加杠杆
        - 做空 = 借入 BTC 卖出，价格下跌时盈利
        - equity = cash + position * price（position 可为负）
    """
    signal = signal.clip(-1.0, 1.0).fillna(0.0)

    if len(signal) != len(candles):
        raise ValueError(
            f"signal 长度 ({len(signal)}) 与 candles 长度 ({len(candles)}) 不匹配"
        )

    cash = initial_capital
    position = 0.0  # BTC 持仓量，正=多头，负=空头
    equity_curve = []
    trades = []

    for i in range(len(candles)):
        if i > 0:
            target_pos = signal.iloc[i - 1]
            exec_price = candles["open"].iloc[i]

            current_btc_value = position * exec_price
            current_equity = cash + current_btc_value
            if current_equity <= 0:
                equity_curve.append(0.0)
                continue
            target_btc_value = current_equity * target_pos
            delta_btc_value = target_btc_value - current_btc_value

            if abs(delta_btc_value) > current_equity * 0.001:
                if delta_btc_value > 0:
                    actual_price = exec_price * (1 + SLIPPAGE_RATE)
                    cost = delta_btc_value * (1 + FEE_RATE)
                    if cost > cash:
                        cost = cash
                        delta_btc_value = cost / (1 + FEE_RATE)
                    btc_amount = delta_btc_value / actual_price
                    cash -= cost
                    position += btc_amount
                    trades.append({
                        "bar": i,
                        "side": "BUY",
                        "price": actual_price,
                        "btc_amount": btc_amount,
                        "value": delta_btc_value,
                    })
                else:
                    actual_price = exec_price * (1 - SLIPPAGE_RATE)
                    btc_to_sell = abs(delta_btc_value) / actual_price
                    revenue = btc_to_sell * actual_price * (1 - FEE_RATE)
                    cash += revenue
                    position -= btc_to_sell
                    trades.append({
                        "bar": i,
                        "side": "SELL",
                        "price": actual_price,
                        "btc_amount": btc_to_sell,
                        "value": abs(delta_btc_value),
                    })

        final_price = candles["close"].iloc[i]
        equity = cash + position * final_price
        equity_curve.append(max(equity, 0.0))

    return {
        "equity_curve": np.array(equity_curve),
        "trades": trades,
        "final_capital": equity_curve[-1] if equity_curve else initial_capital,
        "initial_capital": initial_capital,
    }


# ─────────────────────────────────────────────
# 3. 评估函数
# ─────────────────────────────────────────────

def evaluate(result: dict, strategy_lines: int = 0) -> dict:
    """
    评估回测结果，返回综合评分和详细指标。

    评分规则:
        - 核心指标: Sharpe Ratio
        - max_drawdown > 30% → score 减半
        - num_trades < 10 → score = 0
        - strategy_lines > 100 → score 打 8 折
    """
    equity = result["equity_curve"]
    initial = result["initial_capital"]
    trades = result["trades"]

    # 收益率序列（逐 K 线）
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    # 总收益率
    total_return = (equity[-1] / initial - 1) * 100

    # 年化 Sharpe Ratio（假设 1h K 线，一年约 8760 根）
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(8760)
    else:
        sharpe = 0.0

    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0.0

    # 胜率（按交易对计算）
    buy_sells = _pair_trades(trades)
    num_trades = len(buy_sells)
    if num_trades > 0:
        wins = sum(1 for t in buy_sells if t["pnl"] > 0)
        win_rate = wins / num_trades * 100
    else:
        win_rate = 0.0

    # 买入持有对比
    bh_return = (equity[-1] / initial - 1) if len(equity) > 0 else 0.0

    # ── 综合评分 ──
    # Sharpe × 收益加成：收益越高分数越高，亏损则打折
    return_boost = 1.0 + total_return / 100.0
    if return_boost < 0.1:
        return_boost = 0.1
    score = sharpe * return_boost

    if max_drawdown > 30:
        score *= 0.5
    if num_trades < 10:
        score = 0.0
    if strategy_lines > 100:
        score *= 0.8

    # 防止 NaN
    if not math.isfinite(score):
        score = 0.0

    return {
        "score": round(score, 4),
        "sharpe_ratio": round(sharpe, 4),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 2),
        "num_trades": num_trades,
        "final_capital": round(equity[-1], 2),
        "strategy_lines": strategy_lines,
    }


def _pair_trades(trades: list) -> list:
    """将 BUY/SELL 配对为往返交易（支持多空双向，FIFO）"""
    pairs = []
    buy_stack = []
    sell_stack = []
    for t in trades:
        if t["side"] == "BUY":
            if sell_stack:
                entry = sell_stack.pop(0)
                pairs.append({
                    "buy_price": t["price"],
                    "sell_price": entry["price"],
                    "pnl": (entry["price"] - t["price"]) / entry["price"],
                })
            else:
                buy_stack.append(t)
        elif t["side"] == "SELL":
            if buy_stack:
                entry = buy_stack.pop(0)
                pairs.append({
                    "buy_price": entry["price"],
                    "sell_price": t["price"],
                    "pnl": (t["price"] - entry["price"]) / entry["price"],
                })
            else:
                sell_stack.append(t)
    return pairs


# ─────────────────────────────────────────────
# 4. 工具函数
# ─────────────────────────────────────────────

def count_strategy_lines(filepath: str = "strategy.py") -> int:
    """统计策略文件的有效代码行数（去掉空行和注释）"""
    path = Path(__file__).parent / filepath
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            count += 1
    return count


def format_results(metrics: dict) -> str:
    """格式化评估结果为可读字符串"""
    lines = [
        f"score: {metrics['score']}",
        f"sharpe_ratio: {metrics['sharpe_ratio']}",
        f"total_return: {metrics['total_return_pct']}%",
        f"max_drawdown: {metrics['max_drawdown_pct']}%",
        f"win_rate: {metrics['win_rate_pct']}%",
        f"num_trades: {metrics['num_trades']}",
        f"final_capital: ${metrics['final_capital']}",
        f"strategy_lines: {metrics['strategy_lines']}",
    ]
    return "\n".join(lines)
