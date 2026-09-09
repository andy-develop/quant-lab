#!/usr/bin/env python3
"""文档口径自动刷新 —— 杜绝 HANDOFF/报告文案里的手写陈旧数字。

三件事:
  1. compute_kpis(): 从 data/meta{,_no}/equity.csv + trades.parquet 计算全窗口 KPI
     (收益/最大回撤/夏普√244/笔数/胜率), 作为 HANDOFF 主表格与报告 STRAT_DOC 的
     唯一权威数字来源 (build_report 也 import 本模块, 口径永不分叉)。
  2. CLI 刷新: python scripts/refresh_docs.py 会把最新 KPI 表格写入 HANDOFF.md 的
     <!-- AUTO-KPI --> 锚点区块, 并核对 SCORE_MODE/MAX_HOLD 与文档声明一致。
  3. CI 一致性检查: check_doc_consistency() 校验 build_report 声明的评分模式/持有期
     与 signals.SCORE_MODE / engine.MAX_HOLD 相同, 不一致即抛异常 (CI fail)。

约定: HANDOFF.md 第 3 节主表格必须包在锚点内, 由本脚本维护, 严禁手改:
  <!-- AUTO-KPI:START (refresh_docs.py 生成, 严禁手改) -->
  ...自动内容...
  <!-- AUTO-KPI:END -->
"""
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")

HANDOFF = f"{BASE}/HANDOFF.md"
ANNO_START = "<!-- AUTO-KPI:START"
ANNO_END = "<!-- AUTO-KPI:END -->"

# ---- 文档口径声明 (build_report 亦引用; 改评分/持有期后跑一遍本脚本核对) ----
DOC_SCORE_MODE = "quality_kdj5"
DOC_MAX_HOLD = 10


def compute_kpis() -> dict:
    """全窗口 KPI (开/关双口径)。夏普口径与报告前端一致: 日收益 mean/std*sqrt(244)。"""
    out = {}
    for key, meta_dir in (("on", f"{BASE}/data/meta"), ("off", f"{BASE}/data/meta_no")):
        eq = pd.read_csv(f"{meta_dir}/equity.csv", parse_dates=["date"]).set_index("date")["equity"]
        r = eq.pct_change().dropna()
        trades = pd.read_parquet(f"{meta_dir}/trades.parquet")
        out[key] = {
            "ret": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else 0.0,
            "trades": int(len(trades)),
            "winrate": float((trades["pnl_pct"] > 0).mean()) if len(trades) else 0.0,
            "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        }
    return out


def render_main_table(k: dict | None = None) -> str:
    """HANDOFF 主表格 markdown (含锚点)。"""
    k = k or compute_kpis()

    def row(label, m):
        return (f"| {label} | {m['ret']:+.1%} | **{m['maxdd']:.1%}** | {m['sharpe']:.2f} "
                f"| {m['trades']} | {m['winrate']:.1%} |")

    head = (f"**全窗口回测 ({k['on']['start']} ~ {k['on']['end']}, 初始 100 万, "
            f"`SCORE_MODE={DOC_SCORE_MODE}`)**:\n\n"
            "| 口径 | 收益 | 最大回撤 | 夏普 | 交易 | 胜率 |\n"
            "|---|---|---|---|---|---|")
    note = ("\n\n> 数字由 `scripts/refresh_docs.py` 从 data/meta{,_no} 自动生成, "
            "严禁手改; 夏普口径 √244 与报告前端一致。")
    # note 必须在锚点【内】: 写在锚点外会因替换只覆盖锚点区间而每次运行残留累积
    return (f"{ANNO_START} (refresh_docs.py 生成, 严禁手改) -->\n{head}\n"
            f"{row('开 (默认)', k['on'])}\n{row('关', k['off'])}\n{note}\n{ANNO_END}")


def refresh_handoff() -> bool:
    """替换 HANDOFF.md 中锚点区块; 无锚点则报错退出 (防止锚点被删后静默失效)。"""
    text = open(HANDOFF).read()
    if ANNO_START not in text or ANNO_END not in text:
        sys.exit(f"[refresh_docs] HANDOFF.md 缺少 {ANNO_START}...{ANNO_END} 锚点, 请先手工插入")
    new = re.sub(rf"{re.escape(ANNO_START)}.*?{re.escape(ANNO_END)}",
                 render_main_table().replace("\\", "\\\\"), text, flags=re.S)
    open(HANDOFF, "w").write(new)
    print(f"[refresh_docs] HANDOFF.md 主表格已刷新 ({DOC_SCORE_MODE})")
    return True


def check_doc_consistency() -> None:
    """CI 一致性检查: 代码实际行为 vs 文档声明, 不一致抛异常 (调用方决定是否 fail)。"""
    from signals import SCORE_MODE          # noqa: E402
    import engine                            # noqa: E402
    problems = []
    if SCORE_MODE != DOC_SCORE_MODE:
        problems.append(f"SCORE_MODE 已切到 '{SCORE_MODE}', 但文档声明仍是 '{DOC_SCORE_MODE}' "
                        f"(改 signals.py 后需同步 build_report.DOC_SCORE_MODE / HANDOFF 并跑 refresh_docs)")
    if engine.MAX_HOLD != DOC_MAX_HOLD:
        problems.append(f"engine.MAX_HOLD={engine.MAX_HOLD}, 文档声明 {DOC_MAX_HOLD}")
    if problems:
        raise SystemExit("[refresh_docs] 文档口径不一致!\n  - " + "\n  - ".join(problems))
    print(f"[refresh_docs] 口径一致: SCORE_MODE={SCORE_MODE}, MAX_HOLD={engine.MAX_HOLD}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="文档口径自动刷新/检查")
    ap.add_argument("--check-only", action="store_true", help="只做一致性检查, 不改 HANDOFF")
    a = ap.parse_args()
    check_doc_consistency()
    if not a.check_only:
        refresh_handoff()
        print(render_main_table())
