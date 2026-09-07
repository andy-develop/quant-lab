#!/usr/bin/env python3
"""每日收盘后一键更新: 快照入库 -> 涨停池 -> 信号扫描 -> 回测 -> 报告"""
import os
import sys
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")


def step(name, fn):
    print(f"===== {name} =====", flush=True)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print(f"[WARN] {name} 失败, 继续后续步骤", flush=True)


if __name__ == "__main__":
    import daily_update, signals, engine, build_report
    step("1/4 每日K线快照增量", daily_update.update_kline)
    step("2/4 涨停池/炸板池", daily_update.update_pool)
    step("3/4 信号扫描 + 组合回测", lambda: (signals.run_scan(), engine.run_backtest()))
    step("4/4 生成报告", build_report.main)
    print("全部完成", flush=True)
