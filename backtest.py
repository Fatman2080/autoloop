"""
backtest.py — 回测运行脚本
串联 prepare.py 和 strategy.py，输出评估结果。
用法：uv run backtest.py [--force-download] [--test]
"""

import sys
import traceback
from pathlib import Path

from prepare import (
    download_data,
    split_data,
    backtest,
    evaluate,
    count_strategy_lines,
    format_results,
)


def run(use_test: bool = False, force_download: bool = False):
    # 1. 加载数据
    df = download_data(force=force_download)
    train, val, test = split_data(df)

    # 2. 导入策略（try-catch 防崩溃）
    try:
        from strategy import generate_signals
    except Exception as e:
        print(f"[错误] 无法导入策略: {e}")
        print(f"score: -1")
        return

    # 3. 统计策略代码行数
    strategy_lines = count_strategy_lines()

    # 4. 在训练集上回测
    print("\n" + "=" * 50)
    print("训练集回测")
    print("=" * 50)
    try:
        train_signal = generate_signals(train)
        train_result = backtest(train_signal, train)
        train_metrics = evaluate(train_result, strategy_lines)
        print(format_results(train_metrics))
    except Exception as e:
        print(f"[错误] 训练集回测失败: {e}")
        traceback.print_exc()
        print(f"score: -1")
        return

    # 5. 在验证集上回测（这是 AI 的优化目标）
    print("\n" + "=" * 50)
    print("验证集回测（优化目标）")
    print("=" * 50)
    try:
        val_signal = generate_signals(val)
        val_result = backtest(val_signal, val)
        val_metrics = evaluate(val_result, strategy_lines)
        print(format_results(val_metrics))
    except Exception as e:
        print(f"[错误] 验证集回测失败: {e}")
        traceback.print_exc()
        print(f"score: -1")
        return

    # 6. 测试集（仅在 --test 模式下运行）
    if use_test:
        print("\n" + "=" * 50)
        print("测试集回测（最终评估）")
        print("=" * 50)
        try:
            test_signal = generate_signals(test)
            test_result = backtest(test_signal, test)
            test_metrics = evaluate(test_result, strategy_lines)
            print(format_results(test_metrics))
        except Exception as e:
            print(f"[错误] 测试集回测失败: {e}")
            traceback.print_exc()

    # 7. 输出关键指标行（供 grep 提取）
    print("\n" + "=" * 50)
    print("汇总")
    print("=" * 50)
    print(f"train_score: {train_metrics['score']}")
    print(f"val_score: {val_metrics['score']}")
    print(f"train_return: {train_metrics['total_return_pct']}%")
    print(f"val_return: {val_metrics['total_return_pct']}%")
    print(f"train_sharpe: {train_metrics['sharpe_ratio']}")
    print(f"val_sharpe: {val_metrics['sharpe_ratio']}")
    print(f"train_drawdown: {train_metrics['max_drawdown_pct']}%")
    print(f"val_drawdown: {val_metrics['max_drawdown_pct']}%")


if __name__ == "__main__":
    use_test = "--test" in sys.argv
    force_dl = "--force-download" in sys.argv
    run(use_test=use_test, force_download=force_dl)
