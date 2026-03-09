"""
runner.py — LLM 自动循环优化策略
调用 MiniMax API，自动修改 strategy.py 并回测，无限迭代寻找最优策略。
用法：uv run runner.py [--max-rounds N] [--stale-limit N]
"""

import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

WORKSPACE = Path(__file__).parent
API_URL = "https://api.minimaxi.com/v1/chat/completions"
MODEL = "MiniMax-M2.5"
MAX_COMPLETION_TOKENS = 8192


def load_env():
    env_path = WORKSPACE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_api_key() -> str:
    return os.environ.get("MINIMAX_API_KEY", "")


def read_file(name: str) -> str:
    p = WORKSPACE / name
    return p.read_text() if p.exists() else ""


def write_file(name: str, content: str):
    (WORKSPACE / name).write_text(content)


def git(*args):
    r = subprocess.run(
        ["git"] + list(args),
        cwd=WORKSPACE, capture_output=True, text=True, timeout=30,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def run_backtest() -> str:
    try:
        r = subprocess.run(
            ["uv", "run", "backtest.py"],
            cwd=WORKSPACE, capture_output=True, text=True, timeout=180,
        )
        return r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def parse_metrics(output: str) -> dict | None:
    fields = {}
    for key in [
        "val_score", "train_score",
        "val_return", "train_return",
        "val_sharpe", "train_sharpe",
        "val_drawdown", "train_drawdown",
    ]:
        m = re.search(rf"{key}:\s*([-\d.]+)", output)
        if m:
            fields[key] = m.group(1)
    if "val_score" not in fields:
        return None
    return fields


def extract_code(text: str) -> str | None:
    """从 LLM 回复中提取 python 代码块"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "def generate_signals" in code:
                return code
    return None


def validate_code(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    if "def generate_signals" not in code:
        return False, "缺少 generate_signals 函数"

    forbidden = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped.startswith("import pandas") or stripped.startswith("import numpy"):
                continue
            if stripped.startswith("from pandas") or stripped.startswith("from numpy"):
                continue
            forbidden.append(stripped)
    if forbidden:
        return False, f"禁止的 import: {forbidden}"

    return True, "OK"


def call_llm(api_key: str, prompt: str) -> str | None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "name": "策略研究员", "content": "你是顶级量化交易策略研究员。你只输出完整的 strategy.py 代码，用 ```python ``` 包裹。"},
            {"role": "user", "name": "用户", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=120)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  [API] 429 限流，等待 {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [API] HTTP {resp.status_code}: {resp.text[:300]}")
                return None

            data = resp.json()
            base_resp = data.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                print(f"  [API] 错误: {base_resp}")
                return None

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"  [API] tokens: {usage.get('total_tokens', '?')} "
                  f"(prompt={usage.get('prompt_tokens', '?')}, "
                  f"completion={usage.get('completion_tokens', '?')})")
            return content

        except requests.Timeout:
            print(f"  [API] 超时，重试 {attempt + 1}/3")
            time.sleep(5)
        except Exception as e:
            print(f"  [API] 异常: {e}")
            return None

    return None


def build_prompt(results_tsv: str, current_strategy: str, round_num: int) -> str:
    return f"""## 任务
你是量化交易策略研究员。你的目标是修改 strategy.py 来最大化 BTC/USDT 在验证集上的 val_score（Sharpe Ratio）。

## 严格规则
1. 只能 import pandas 和 numpy
2. 不能使用未来数据：第 i 行信号只依赖第 0~i 行数据，用 .rolling()/.shift() 确保
3. 函数签名 `generate_signals(candles: pd.DataFrame) -> pd.Series` 不可改
4. 输出范围 -1.0~1.0（正=做多，负=做空）
5. 策略代码不超过 100 行（否则打折）

## 评分机制
- 核心：Sharpe Ratio
- 最大回撤 > 30% → score 减半
- 交易次数 < 10 → score = 0
- 代码 > 100 行 → score × 0.8

## 数据列
价格：timestamp, open, high, low, close, volume
衍生品（仅后30%数据有值）：funding_rate, open_interest, liq_long_usd, liq_short_usd, liq_total_usd, long_short_ratio

## 历史迭代结果（第 {round_num} 轮）
```
{results_tsv}
```

## 当前最佳策略
```python
{current_strategy}
```

## 要求
1. 分析历史结果，找出规律（什么有效、什么无效）
2. 提出一个具体的改进方向（不要重复已失败的方向）
3. 每次只改一个变量，方便判断效果
4. 输出完整的 strategy.py 代码，用 ```python ``` 包裹
5. 在代码的 docstring 中简述你的改动和理由"""


def append_results_tsv(round_num: int, desc: str, metrics: dict, status: str):
    line = (
        f"{round_num}\t{desc}\t{metrics.get('train_score', '?')}\t"
        f"{metrics.get('val_score', '?')}\t"
        f"{metrics.get('train_return', '?')}%\t{metrics.get('val_return', '?')}%\t"
        f"{metrics.get('train_drawdown', '?')}%\t{metrics.get('val_drawdown', '?')}%\t"
        f"{status}\n"
    )
    with open(WORKSPACE / "results.tsv", "a") as f:
        f.write(line)


def get_strategy_description(code: str) -> str:
    """从 docstring 中提取策略描述"""
    m = re.search(r'""".*?策略[：:](.*?)(?:\n\s*-|\n\s*""")', code, re.DOTALL)
    if m:
        return m.group(1).strip()[:60]
    m = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
        for line in lines:
            if "改动" in line or "策略" in line or "变化" in line:
                return line[:60]
    return f"LLM-round"


def main():
    load_env()
    api_key = get_api_key()
    if not api_key:
        print("[错误] 未找到 MINIMAX_API_KEY，请在 .env 中设置")
        sys.exit(1)

    max_rounds = 200
    stale_limit = 30
    for arg in sys.argv[1:]:
        if arg.startswith("--max-rounds="):
            max_rounds = int(arg.split("=")[1])
        elif arg.startswith("--stale-limit="):
            stale_limit = int(arg.split("=")[1])

    results_tsv = read_file("results.tsv")
    current_best_line = None
    for line in results_tsv.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            try:
                score = float(parts[3])
                if current_best_line is None or score > current_best_line:
                    current_best_line = score
            except ValueError:
                pass
    best_val_score = current_best_line if current_best_line is not None else -999.0

    last_round = 20
    for line in results_tsv.strip().splitlines():
        parts = line.split("\t")
        if parts and parts[0].isdigit():
            last_round = max(last_round, int(parts[0]))
    start_round = last_round + 1

    stale_count = 0
    total_tokens = 0

    print(f"\n{'='*60}")
    print(f"  AutoLoop Runner — MiniMax {MODEL}")
    print(f"  当前最佳 val_score: {best_val_score}")
    print(f"  起始轮次: {start_round}")
    print(f"  最大轮次: {max_rounds} | 停滞上限: {stale_limit}")
    print(f"{'='*60}\n")

    for rnd in range(start_round, start_round + max_rounds):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'─'*50}")
        print(f"第 {rnd} 轮  [{ts}]  最佳={best_val_score}  停滞={stale_count}/{stale_limit}")
        print(f"{'─'*50}")

        results_tsv = read_file("results.tsv")
        current_strategy = read_file("strategy.py")

        print("  [1/5] 调用 LLM...")
        prompt = build_prompt(results_tsv, current_strategy, rnd)
        response = call_llm(api_key, prompt)
        if not response:
            print("  LLM 调用失败，跳过")
            stale_count += 1
            time.sleep(10)
            continue

        print("  [2/5] 解析代码...")
        code = extract_code(response)
        if not code:
            print("  未提取到有效代码，跳过")
            print(f"  回复前200字: {response[:200]}")
            stale_count += 1
            continue

        valid, msg = validate_code(code)
        if not valid:
            print(f"  代码校验失败: {msg}")
            stale_count += 1
            continue

        desc = get_strategy_description(code)
        print(f"  策略描述: {desc}")

        old_strategy = current_strategy
        write_file("strategy.py", code)
        git("add", "-A")
        git("commit", "-m", f"round-{rnd}: {desc}")

        print("  [3/5] 运行回测...")
        output = run_backtest()
        if "TIMEOUT" in output:
            print("  回测超时，回滚")
            git("revert", "HEAD", "--no-edit")
            stale_count += 1
            continue

        metrics = parse_metrics(output)
        if not metrics:
            print("  回测结果解析失败，回滚")
            print(f"  输出: {output[-500:]}")
            git("revert", "HEAD", "--no-edit")
            stale_count += 1
            continue

        new_val_score = float(metrics["val_score"])
        print(f"  [4/5] 结果: val_score={new_val_score}  "
              f"train={metrics.get('train_score', '?')}  "
              f"return={metrics.get('val_return', '?')}%  "
              f"drawdown={metrics.get('val_drawdown', '?')}%")

        if new_val_score > best_val_score:
            print(f"  [5/5] ✅ 提升! {best_val_score} → {new_val_score}")
            best_val_score = new_val_score
            stale_count = 0
            append_results_tsv(rnd, desc, metrics, "保留✔")
        else:
            print(f"  [5/5] ❌ 未提升 ({new_val_score} <= {best_val_score})，回滚")
            git("revert", "HEAD", "--no-edit")
            append_results_tsv(rnd, desc, metrics, "回滚")
            stale_count += 1

        if stale_count >= stale_limit:
            print(f"\n连续 {stale_limit} 轮无提升，自动停止。")
            break

        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"  运行结束")
    print(f"  最终最佳 val_score: {best_val_score}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
