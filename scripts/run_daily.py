#!/usr/bin/env python3
"""每日收盘后一键更新: 快照入库 -> 涨停池 -> 信号扫描 -> 回测 -> 报告"""
import os
import sys
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")


def step(name: str, fn, fatal: bool = False) -> None:
    """fatal=True 的步骤失败时中止整个流程:
    数据链路(快照/信号/回测/报告)在损坏或过期状态上继续跑, 会产生静默错误的
    输出并上传误导性报告 —— 宁可当天失败留旧数据, 也不带病跑下游。"""
    print(f"===== {name} =====", flush=True)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        if fatal:
            print(f"[FATAL] {name} 失败, 中止流程", file=sys.stderr, flush=True)
            raise SystemExit(1)
        print(f"[WARN] {name} 失败, 继续后续步骤", flush=True)


if __name__ == "__main__":
    import daily_update, lgbm_rank, signals, engine, build_report
    step("1/5 每日K线快照增量", daily_update.update_kline, fatal=True)
    step("2/5 涨停池/炸板池", daily_update.update_pool)   # 池子缺失不影响信号, 可降级
    def scan_and_backtest():
        signals.run_scan()
        # 双口径: 开仓位控制(默认展示) + 关仓位控制, 分别落盘
        engine.run_backtest(use_regime=True, out_dir=f"{BASE}/data/meta")
        engine.run_backtest(use_regime=False, out_dir=f"{BASE}/data/meta_no")
    step("3/5 信号扫描 + 组合回测(动量双口径)", scan_and_backtest, fatal=True)
    def blackbox():
        # 量化黑盒: LightGBM 排序模型打分(walk-forward, 增量) + 同选股条件回测双口径。
        # 非 fatal: 模型链路故障不拖垮动量主报告, 黑盒页沿用最近一次成功数据。
        lgbm_rank.update_scores()
        sig_bb = f"{BASE}/data/meta/signals_bb.parquet"
        signals.run_scan(score_mode="lgbm", out_file=sig_bb, strat="blackbox")
        engine.run_backtest(use_regime=True, out_dir=f"{BASE}/data/meta_bb", sig_file=sig_bb)
        engine.run_backtest(use_regime=False, out_dir=f"{BASE}/data/meta_no_bb", sig_file=sig_bb)
    step("4/5 量化黑盒: 模型打分 + 回测(双口径)", blackbox)
    step("5/5 生成报告", build_report.main, fatal=True)   # 报告失败时中止, 避免上传过期报告
    print("全部完成", flush=True)
