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
    def scan_and_backtest():
        signals.run_scan()
        # 双口径: 开仓位控制(默认展示) + 关仓位控制, 分别落盘
        engine.run_backtest(use_regime=True, out_dir=f"{BASE}/data/meta")
        engine.run_backtest(use_regime=False, out_dir=f"{BASE}/data/meta_no")
    step("3/4 信号扫描 + 组合回测(双口径)", scan_and_backtest)
    step("4/4 生成报告", build_report.main)
    print("全部完成", flush=True)
