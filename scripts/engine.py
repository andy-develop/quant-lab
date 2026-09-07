import os
#!/usr/bin/env python3
"""日线组合回测引擎。A股规则内建:
  - T日收盘出信号 -> T+1开盘成交
  - 开盘涨停(开盘价>=涨停价)无法买入 -> 放弃
  - 开盘跌停无法卖出 -> 顺延至下一开盘
  - 停牌(当日无K线)跳过
  - 成本: 佣金万1双边 + 印花税0.05%(卖出) + 滑点0.1%单边
  - 仓位控制: 每日最多买入3只; 总仓位锚定上证指数买卖点状态机
    (Z0空仓0%且清仓 / Z1轻仓30% / Z2半仓50% / Z3重仓100%), 见 signals.market_regime
  - T+1: 买入次日不可卖(引擎天然满足: 卖出信号在收盘判定, 最早次日开盘执行)
会计口径: 前复权价格(自动含分红除权), 交易流水展示不复权价。
"""
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)

MAX_POS = 10
MAX_BUY_PER_DAY = 3      # 每日新开仓上限
START_CAP = 1_000_000.0
STOP_PCT = 0.92          # 止损: 收盘 <= 买入价*0.92
MAX_HOLD = 10            # 最长持有交易日
MIN_HOLD_BREAK = 3       # 趋势破位最小持有
MIN_HOLD_SHRINK = 2      # 放量滞涨最小持有
COMM, STAMP, SLIP = 0.0001, 0.0005, 0.001
EXIT_PRIORITY = ["stop_loss", "trend_break", "vol_shrink", "expired"]


def limit_pct(code):
    return 0.20 if code[:2] in ("30", "68") else 0.10


def load_wide():
    fl = pd.read_parquet(f"{BASE}/data/meta/flags_long.parquet")
    fl["date"] = pd.to_datetime(fl["date"])
    piv = {}
    for col in ["open", "close", "raw_close", "raw_open", "break_c", "shrink"]:
        p = fl.pivot(index="date", columns="code", values=col)
        piv[col] = p
    return piv


def load_signals_wide():
    sig = pd.read_parquet(f"{BASE}/data/meta/signals.parquet")
    sig["date"] = pd.to_datetime(sig["date"])
    # 同一(code,date)去重取分最高
    sig = sig.sort_values("score", ascending=False).drop_duplicates(["code", "date"])
    out = {}
    for strat, g in sig.groupby("strategy"):
        out[strat] = g.pivot(index="date", columns="code", values="score")
    names = sig.drop_duplicates("code").set_index("code")["name"]
    return out, names


def run_backtest(start=None, min_buy_ratio=0.0, quick_fail=0.0, de_risk_pt=0.0, regime_file=None):
    piv = load_wide()
    sig_w, names = load_signals_wide()
    cal = piv["close"].index
    cal = cal[cal >= pd.Timestamp(start)] if start else cal
    start = start or str(cal[0].date())

    qo, qc = piv["open"], piv["close"]
    ro, rc = piv["raw_open"], piv["raw_close"]
    # 涨跌停判定: 用前复权比值 (开盘/昨收 - 1 对比涨跌幅), 容差0.002 吸收除权与精度差
    # (不能拿前复权价对比不复权涨停价——量纲不一致会让一字板全部漏判)
    prev_qc = qc.shift(1)
    pct = pd.Series({c: limit_pct(c) for c in qc.columns})
    pct_frame = pd.DataFrame({c: pct[c] for c in qc.columns}, index=qc.index)
    open_ret = qo / prev_qc - 1
    no_buy = open_ret >= (pct_frame - 0.002)          # 开盘涨停/接近一字 -> 买不进
    no_sell = open_ret <= -(pct_frame - 0.002)        # 开盘跌停 -> 卖不出

    strat_break = {"momentum": "break_c"}

    # 大盘仓位状态机 (买卖点, 默认锚定上证指数; regime_file 可替换如中证1000)
    # 关键: 状态由 T 日收盘计算, T 日开盘不可见 -> 必须平移一日, 用 T-1 收盘状态驱动 T 日开盘动作 (防未来函数)
    regime_file = regime_file or f"{BASE}/data/meta/market_regime.parquet"
    regime = pd.read_parquet(regime_file)
    regime["date"] = pd.to_datetime(regime["date"])
    regime = regime.set_index("date")["target_ratio"].shift(1)

    cash = START_CAP
    invested = 0.0        # 持仓市值 (随卖出/买入即时维护, 用于预算控制)
    positions = {}   # code -> dict
    trades, holdings_daily = [], []
    equity = []
    pending_sells = {}   # code -> reason
    cal_pos = {d: i for i, d in enumerate(cal)}

    for day in cal:
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
            fee = proceeds * (COMM + STAMP)
            cash += proceeds - fee
            invested -= proceeds - fee
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
        # 仓位控制: 每日最多 MAX_BUY_PER_DAY 只; 总市值不得超过 max_invest (状态机目标仓位)
        prev_day = cal[cal_pos[day] - 1] if cal_pos[day] > 0 else None
        cands = []
        if prev_day is not None and ratio >= max(min_buy_ratio, 0.001):
            for strat, sw in sig_w.items():
                if prev_day in sw.index:
                    row = sw.loc[prev_day].dropna()
                    for code, sc in row.items():
                        cands.append((float(sc), strat, code))
        cands.sort(reverse=True)
        held = set(positions) | set(pending_sells)
        free = MAX_POS - len(positions)
        n_try = 0
        for rank_c, (sc, strat, code) in enumerate(cands, 1):
            if free <= 0 or n_try >= MAX_BUY_PER_DAY:
                break
            if invested >= max_invest - 1000:   # 预算用尽 (1000元容差)
                break
            if code in held:
                continue
            px = qo.at[day, code] if code in qo.columns else np.nan
            if np.isnan(px):
                continue                      # 停牌
            if code in no_buy.columns and bool(no_buy.at[day, code]):
                continue                      # 开盘涨停买不进
            slot = min(equity[-1][1] / MAX_POS if equity else START_CAP / MAX_POS,
                       cash / max(free - n_try, 1),
                       max_invest - invested)  # 不超目标仓位预算
            shares = int(slot / (px * 100)) * 100
            if shares < 100:
                continue
            cost = shares * px * (1 + SLIP)
            fee = cost * COMM
            cash -= cost + fee
            invested += cost + fee
            n_try += 1
            positions[code] = {
                "name": names.get(code, code), "strategy": strat,
                "entry_date": day, "entry_qfq": px, "entry_px_raw": ro.at[day, code],
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
            val = pos["shares"] * (cl if not np.isnan(cl) else pos["entry_qfq"])
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
            if quick_fail and pos["hold_days"] == 1 and pnl_pct <= quick_fail:
                reason = "quick_fail"      # 快速认错: 首日收盘浮亏超阈值, 次日开盘退出
            elif not np.isnan(cl) and cl <= pos["entry_qfq"] * STOP_PCT:
                reason = "stop_loss"
            elif pos["hold_days"] >= MAX_HOLD:
                reason = "expired"
            elif pos["hold_days"] >= MIN_HOLD_BREAK:
                bcol = strat_break[pos["strategy"]]
                bv = piv[bcol].at[day, code] if code in piv[bcol].columns else False
                if bool(bv):
                    reason = "trend_break"
            if reason is None and pos["hold_days"] >= MIN_HOLD_SHRINK:
                sv = piv["shrink"].at[day, code] if code in piv["shrink"].columns else False
                if bool(sv):
                    reason = "vol_shrink"
            if reason:
                pending_sells[code] = reason

        # ---------- 仓位回归 (实验, de_risk_pt>0 启用): 实际仓位超目标 de_risk_pt -> 次日开盘卖最弱 ----------
        if de_risk_pt > 0 and ratio > 0:
            def _val(p):
                cl = p["last_close"]
                return p["shares"] * (cl if not np.isnan(cl) else p["entry_qfq"])
            mv = total - cash
            budget = total * ratio
            if mv > budget * (1 + de_risk_pt):
                for code, pos in sorted(positions.items(), key=lambda kv: _val(kv[1]) / kv[1]["cost"]):
                    if mv <= budget:
                        break
                    if code in pending_sells:
                        continue
                    pending_sells[code] = "de_risk"
                    mv -= _val(pos)

        equity.append((day, total))
        if snap:
            for s in snap:
                s["date"] = day
            holdings_daily.extend(snap)

    eq = pd.DataFrame(equity, columns=["date", "equity"]).set_index("date")
    tr = pd.DataFrame(trades)
    hd = pd.DataFrame(holdings_daily)
    eq.to_csv(f"{BASE}/data/meta/equity.csv")
    # 待卖清单: 最后交易日收盘时已挂卖出、将于下一开盘执行的持仓
    pd.DataFrame([{"code": c, "reason": r} for c, r in pending_sells.items()]) \
        .to_parquet(f"{BASE}/data/meta/pending_sells.parquet", index=False)
    if len(tr):
        tr.to_parquet(f"{BASE}/data/meta/trades.parquet", index=False)
        print(f"交易 {len(tr)} 笔, 胜率 {(tr['pnl_pct']>0).mean():.1%}, "
              f"总收益 {eq['equity'].iloc[-1]/START_CAP-1:+.1%}", flush=True)
    else:
        print("无交易", flush=True)
    if len(hd):
        hd.to_parquet(f"{BASE}/data/meta/holdings.parquet", index=False)
    return eq, tr, hd


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else None
    run_backtest(start)
