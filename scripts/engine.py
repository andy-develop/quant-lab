#!/usr/bin/env python3
"""日线组合回测引擎。A股规则内建:
  - T日收盘出信号 -> T+1开盘成交
  - 开盘涨停(开盘价>=涨停价)无法买入 -> 放弃
  - 开盘跌停无法卖出 -> 顺延至下一开盘
  - 停牌(当日无K线)跳过
  - 成本: 佣金万1双边(最低5元) + 印花税0.05%(卖出) + 滑点0.1%单边
  - 仓位: 每日最多买入3只, 最多同时持有10只; 等权分仓(总权益/10)
  - T+1: 买入次日不可卖(引擎天然满足: 卖出信号在收盘判定, 最早次日开盘执行)
会计口径: 后复权价格(hfq, 连续且历史值永久冻结)做信号/涨跌停判定/估值/现金流;
交易流水展示不复权价。hfq 与 qfq 的区间收益率完全一致(复权因子在比值中约掉)。
预算控制: 持仓市值实时按最近收盘价重算(不维护增量记账, 避免已实现盈亏泄漏)。

双模式 (2026-09-08): use_regime 参数切换
  True  = 大盘仓位状态机 (Z0空仓0%且清仓 / Z1轻仓30% / Z2半仓50% / Z3重仓100%,
          锚定上证指数买卖点, 见 signals.market_regime) —— 报告默认展示
  False = 始终满仓, 无大盘择时
A/B 参考(全期2023-09~2026-09, SCORE_MODE=quality_kdj5, 由 scripts/refresh_docs.py
从 data/meta{,_no} 自动生成, 最新数字以 HANDOFF.md 锚点区块为准):
  开 +114.1%/夏普1.44/回撤-13.1%/460笔;  关 +52.1%/夏普0.60/回撤-37.4%/849笔。
口径演变提示: 上述数字随评分(纯动量→quality→quality_kdj5)与 ST 未来函数修复而变,
旧文档中的 开+32.2%/关-78.4% 为纯动量+ST修复前口径, 已失效 —— 勿引用。
注: 趋势破位(破MA20)退出规则已于 2026-09-07 移除 —— A/B 证实纯负贡献
(74笔合计-17.7万, 删除后全期 +22.5% -> +32.2%, 夏普 0.39 -> 0.49)。
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)

MAX_POS = 10
MAX_BUY_PER_DAY = 3      # 每日新开仓上限
START_CAP = 1_000_000.0
STOP_PCT = 0.92          # 止损: 收盘 <= 买入价*0.92
MAX_HOLD = 10            # 最长持有交易日
MIN_HOLD_SHRINK = 2      # 放量滞涨最小持有
COMM, STAMP, SLIP = 0.0001, 0.0005, 0.001
MIN_FEE = 5.0            # 单笔佣金最低 5 元
LOT_SIZE = 100           # A股整手 (股)
LIMIT_TOL = 0.002        # 涨跌停判定容差 (吸收复权/精度差)
BUDGET_TOL = 1000.0      # 目标仓位预算判定容差 (元)
MIN_REGIME_RATIO = 0.001 # 状态机目标仓位低于此视为空仓, 不开新仓


def limit_pct(code: str) -> float:
    c = code.split(".", 1)[-1]            # "1.300750"/"sz.300750" -> "300750"
    return 0.20 if c[:2] in ("30", "68") else 0.10


def buy_fee(amount: float) -> float:
    return max(amount * COMM, MIN_FEE)


def sell_fee(amount: float) -> float:
    return max(amount * COMM, MIN_FEE) + amount * STAMP   # 印花税无下限


def load_wide() -> dict[str, pd.DataFrame]:
    fl = pd.read_parquet(f"{BASE}/data/meta/flags_long.parquet")
    fl["date"] = pd.to_datetime(fl["date"])
    piv = {}
    for col in ["open", "close", "raw_close", "raw_open", "shrink"]:
        p = fl.pivot(index="date", columns="code", values=col)
        piv[col] = p
    return piv


def load_signals_wide(sig_file: str | None = None) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    sig = pd.read_parquet(sig_file or f"{BASE}/data/meta/signals.parquet")
    sig["date"] = pd.to_datetime(sig["date"])
    # 同一(code,date)去重取分最高
    sig = sig.sort_values("score", ascending=False).drop_duplicates(["code", "date"])
    out = {}
    for strat, g in sig.groupby("strategy"):
        out[strat] = g.pivot(index="date", columns="code", values="score")
    names = sig.drop_duplicates("code").set_index("code")["name"]
    return out, names


def run_backtest(start: str | None = None, use_regime: bool = True,
                 out_dir: str | None = None, max_hold: int = MAX_HOLD,
                 sig_file: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    piv = load_wide()
    sig_w, names = load_signals_wide(sig_file)
    cal = piv["close"].index
    cal = cal[cal >= pd.Timestamp(start)] if start else cal
    start = start or str(cal[0].date())
    # 默认目录随 use_regime 走: 开->meta / 关->meta_no (此前恒为 meta, 关口径漏传参即覆盖生产数据)
    out_dir = out_dir or f"{BASE}/data/{'meta' if use_regime else 'meta_no'}"
    os.makedirs(out_dir, exist_ok=True)

    qo, qc = piv["open"], piv["close"]        # hfq 口径 (连续, 涨跌停判定/成交/估值)
    ro, rc = piv["raw_open"], piv["raw_close"]  # 不复权 (展示价)
    # 涨跌停判定: 用后复权比值 (开盘/昨收 - 1 对比涨跌幅), 容差0.002 吸收精度差
    # (hfq 同日线性缩放, 比值即真实涨跌幅, 且跨除权日连续;
    #  ffill 修复停牌日 NaN -> 复牌首日跳空可被正确识别)
    prev_qc = qc.ffill().shift(1)
    pct = pd.Series({c: limit_pct(c) for c in qc.columns})
    pct_frame = pd.DataFrame({c: pct[c] for c in qc.columns}, index=qc.index)
    open_ret = qo / prev_qc - 1
    no_buy = open_ret >= (pct_frame - LIMIT_TOL)          # 开盘涨停/接近一字 -> 买不进
    no_sell = open_ret <= -(pct_frame - LIMIT_TOL)        # 开盘跌停 -> 卖不出

    # 大盘仓位状态机 (use_regime=False 时不启用, 始终满仓)
    # 关键: 状态由 T 日收盘计算, T 日开盘不可见 -> 必须平移一日, 用 T-1 收盘状态驱动 T 日开盘动作 (防未来函数)
    regime = None
    if use_regime:
        _reg = pd.read_parquet(f"{BASE}/data/meta/market_regime.parquet")
        _reg["date"] = pd.to_datetime(_reg["date"])
        regime = _reg.set_index("date")["target_ratio"].shift(1)

    cash = START_CAP
    positions = {}   # code -> dict
    trades, holdings_daily = [], []
    equity = []
    pending_sells = {}   # code -> reason
    cal_pos = {d: i for i, d in enumerate(cal)}

    for day in cal:
        ratio = 1.0
        if use_regime:
            if day in regime.index:
                ratio = float(regime.at[day])
            else:
                ratio = float(regime.asof(day)) if len(regime) else 1.0
            max_invest = (equity[-1][1] if equity else START_CAP) * ratio

            # ---------- 逃顶: Z0 空仓 -> 全部挂卖出 ----------
            if ratio <= 0.0:
                for code in positions:
                    if code not in pending_sells:
                        pending_sells[code] = "market_exit"

        # ---------- 开盘: 先卖后买 ----------
        for code in list(pending_sells):
            reason = pending_sells[code]
            px = qo.at[day, code] if code in qo.columns else np.nan
            if np.isnan(px):
                continue                      # 停牌, 顺延
            if code in no_sell.columns and bool(no_sell.at[day, code]):
                continue                      # 开盘跌停卖不出, 顺延
            pos = positions.pop(code)
            proceeds = pos["shares"] * px * (1 - SLIP)
            fee = sell_fee(proceeds)
            cash += proceeds - fee
            exit_raw = ro.at[day, code]
            pnl_cny = proceeds - fee - pos["cost"]
            trades.append({
                "code": code, "name": pos["name"], "strategy": pos["strategy"],
                "entry_date": pos["entry_date"], "exit_date": day,
                "entry_px": pos["entry_px_raw"], "exit_px": exit_raw,
                "shares": pos["shares"], "pnl_pct": pnl_cny / pos["cost"],
                "pnl_cny": pnl_cny, "reason": reason,
                "hold_days": pos["hold_days"],
                "buy_rank": pos.get("buy_rank", 0),
            })
            del pending_sells[code]

        # 买入: 【前一交易日收盘】的信号 -> 今日开盘执行 (防未来函数)
        # 开模式: 持仓市值 mv 实时按最近收盘价重算, 不得超过 max_invest (状态机目标仓位)
        prev_day = cal[cal_pos[day] - 1] if cal_pos[day] > 0 else None
        mv = 0.0
        if use_regime:
            for pos_ in positions.values():
                cl_ = pos_.get("last_close", np.nan)
                mv += pos_["shares"] * (cl_ if not np.isnan(cl_) else pos_["entry_adj"])
        cands = []
        if prev_day is not None and ratio >= MIN_REGIME_RATIO:
            for strat, sw in sig_w.items():
                if prev_day in sw.index:
                    row = sw.loc[prev_day].dropna()
                    for code, sc in row.items():
                        cands.append((float(sc), strat, code))
        cands.sort(reverse=True)
        held = set(positions) | set(pending_sells)
        free = MAX_POS - len(positions)     # 剩余空位 (每买一只减 1, 兼作现金分摊分母)
        n_try = 0
        for rank_c, (sc, strat, code) in enumerate(cands, 1):
            if free <= 0 or n_try >= MAX_BUY_PER_DAY:
                break
            if use_regime and mv >= max_invest - BUDGET_TOL:   # 预算用尽 (BUDGET_TOL 容差)
                break
            if code in held:
                continue
            px = qo.at[day, code] if code in qo.columns else np.nan
            if np.isnan(px):
                continue                      # 停牌
            if code in no_buy.columns and bool(no_buy.at[day, code]):
                continue                      # 开盘涨停买不进
            slot = min(equity[-1][1] / MAX_POS if equity else START_CAP / MAX_POS,
                       cash / max(free, 1))
            if use_regime:
                slot = min(slot, max_invest - mv)    # 不超目标仓位预算
            shares = int(slot / (px * LOT_SIZE)) * LOT_SIZE   # A股整手
            if shares < LOT_SIZE:
                continue
            cost = shares * px * (1 + SLIP)
            fee = buy_fee(cost)
            cash -= cost + fee
            if use_regime:
                mv += cost + fee
            n_try += 1
            positions[code] = {
                "name": names.get(code, code), "strategy": strat,
                "entry_date": day, "entry_adj": px, "entry_px_raw": ro.at[day, code],
                "shares": shares, "cost": cost + fee, "hold_days": 0,
                "buy_rank": rank_c, "buy_score": sc,
            }
            held.add(code)
            free -= 1

        # ---------- 收盘: 估值 + 退出判定 ----------
        total = cash
        snap = []
        for code, pos in positions.items():
            cl = qc.at[day, code] if code in qc.columns else np.nan
            val = pos["shares"] * (cl if not np.isnan(cl) else pos["entry_adj"])
            pos["hold_days"] += 1
            pos["last_close"] = cl
            total += val
            pnl_pct = val / pos["cost"] - 1
            snap.append({"code": code, "name": pos["name"], "strategy": pos["strategy"],
                         "entry_date": pos["entry_date"], "entry_px": pos["entry_px_raw"],
                         "shares": pos["shares"], "value": val, "pnl_pct": pnl_pct,
                         "hold_days": pos["hold_days"]})
            # 退出判定(收盘, 顺延明日开盘)
            if code in pending_sells:
                continue
            reason = None
            if not np.isnan(cl) and cl <= pos["entry_adj"] * STOP_PCT:
                reason = "stop_loss"
            elif pos["hold_days"] >= max_hold:
                reason = "expired"
            elif pos["hold_days"] >= MIN_HOLD_SHRINK:
                sv = piv["shrink"].at[day, code] if code in piv["shrink"].columns else False
                if bool(sv):
                    reason = "vol_shrink"
            if reason:
                pending_sells[code] = reason

        equity.append((day, total))
        if snap:
            for s in snap:
                s["date"] = day
            holdings_daily.extend(snap)

    eq = pd.DataFrame(equity, columns=["date", "equity"]).set_index("date")
    tr = pd.DataFrame(trades)
    hd = pd.DataFrame(holdings_daily)
    eq.to_csv(f"{out_dir}/equity.csv")
    # 待卖清单: 最后交易日收盘时已挂卖出、将于下一开盘执行的持仓
    pd.DataFrame([{"code": c, "reason": r} for c, r in pending_sells.items()],
                 columns=["code", "reason"]) \
        .to_parquet(f"{out_dir}/pending_sells.parquet", index=False)
    if len(tr):
        tr.to_parquet(f"{out_dir}/trades.parquet", index=False)
        print(f"交易 {len(tr)} 笔, 胜率 {(tr['pnl_pct']>0).mean():.1%}, "
              f"总收益 {eq['equity'].iloc[-1]/START_CAP-1:+.1%}", flush=True)
    else:
        print("无交易", flush=True)
    if len(hd):
        hd.to_parquet(f"{out_dir}/holdings.parquet", index=False)
    return eq, tr, hd


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else None
    run_backtest(start)
