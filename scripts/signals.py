#!/usr/bin/env python3
"""信号层: 动量轮动策略 + 共享退出标志 + 大盘仓位状态机。全部基于 T 日收盘信息, T+1 执行。

价格口径: 信号指标全部基于【后复权 hfq】序列 (历史值永久冻结, 可复现;
与 qfq 的区间收益一致, 但 qfq 随最新价整体缩放会破坏可复现性, 已弃用)。
成交量/成交额用不复权真实值。

策略C momentum     动量轮动: 20日收益率横截面Top 5%, 多头排列

大盘仓位状态机 (买卖点, 锚定上证指数):
  Z0 空仓0%(清仓) / Z1 轻仓30% / Z2 半仓50% / Z3 重仓100%
引擎侧按 use_regime 开关启用 (2026-09-08 起报告页支持一键切换双口径)。

退出标志(T收盘判定, T+1开盘执行):
  shrink : 放量滞涨 vol>=2*vol_ma5[-1] 且 收阴
引擎侧另有: 止损-8%、持有满10日、(开模式) Z0 逃顶清仓。
注: 趋势破位(破MA20)已于 2026-09-07 移除 —— A/B 证实纯负贡献。
"""
import glob, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
START = "2023-09-01"


def load_klines(kind):
    """kind: raw / hfq -> 分片 + 增量 + 除权修复, 合并为长表"""
    files = sorted(glob.glob(f"{BASE}/data/kline/{kind}_*.parquet"))
    files += sorted(glob.glob(f"{BASE}/data/kline/incremental/{kind}_*.parquet"))
    if not files:
        raise SystemExit(f"没有找到 {kind} 分片, 先运行 backfill_kline.py")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fixs = sorted(glob.glob(f"{BASE}/data/kline/fixup/{kind}_*.parquet"))
    if fixs:
        fix = pd.concat([pd.read_parquet(f) for f in fixs], ignore_index=True)
        df = df[~df["code"].isin(set(fix["code"]))]
        df = pd.concat([df, fix], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # 统一代码格式: baostock 用 sh.600000, 腾讯/股票池用 1.600000
    df["code"] = df["code"].str.replace("sh.", "1.", regex=False).str.replace("sz.", "0.", regex=False)
    return df.drop_duplicates(["code", "date"]).sort_values(["code", "date"])


def groll(s, w, fn):
    return s.groupby(level=0, sort=False).transform(lambda x: getattr(x.rolling(w, min_periods=w), fn)())


def gshift(s, n):
    return s.groupby(level=0, sort=False).transform(lambda x: x.shift(n))


def build_indicators(hfq, raw):
    """在 (code,date) MultiIndex 上计算全部指标。
    hfq: 后复权 OHLC —— 动量/均线等比值类指标, 历史值永久冻结、可复现;
         与 qfq 计算的区间收益完全一致(复权因子在比值中约掉)。
    raw: 不复权 —— 真实成交量/成交额/展示价。
    """
    q = hfq.set_index(["code", "date"]).sort_index()
    r = raw.set_index(["code", "date"]).sort_index()
    df = pd.DataFrame(index=q.index)
    df["close"] = q["close"]
    df["open"] = q["open"]
    df["high"] = q["high"]
    df["low"] = q["low"]
    df["volume"] = r["volume"]          # 量能从 raw 取 (量本就不复权)
    df["raw_close"] = r["close"]

    # 均线与前值
    for w in [5, 10, 20, 50, 60]:
        df[f"ma{w}"] = groll(df["close"], w, "mean")
    df["vol_ma5"] = groll(df["volume"], 5, "mean")
    df["vol_ma20"] = groll(df["volume"], 20, "mean")
    df["prev_vol_ma5"] = gshift(df["vol_ma5"], 1)
    df["prev_vol_ma20"] = gshift(df["vol_ma20"], 1)
    df["ret"] = df["close"] / gshift(df["close"], 1) - 1
    df["ret20"] = df["close"] / gshift(df["close"], 20) - 1
    df["ret120"] = df["close"] / gshift(df["close"], 120) - 1
    df["ma60_rising"] = df["ma60"] > gshift(df["ma60"], 20)
    # 流动性: 优先用真实成交额(baostock raw 分片/腾讯快照均含 amount);
    # 缺失时退化为 收盘*量(手)*100 近似
    if "amount" in r.columns:
        df["amt"] = r["amount"].fillna(df["close"] * df["volume"] * 100)
    else:
        df["amt"] = df["close"] * df["volume"] * 100
    df["amt20"] = groll(df["amt"], 20, "mean")
    r_open = r["open"].reindex(df.index)
    df["raw_open"] = r_open

    return df


def market_regime(index_file=None, out_file=None):
    """买卖点仓位状态机 -> 每日目标仓位比例 (默认锚定上证指数)
    参考: 用户腾讯文档《买卖点》(docs.qq.com/doc/DWGdJeWR0amRjc3ZC)
    Z0 空仓 / Z1 轻仓30% / Z2 半仓50% / Z3 重仓100%
    注: 2026-09-07 A/B 复核 —— 状态机(含 Z0 逃顶与阶梯预算)是策略核心风控,
    开启 +32.2%/夏普0.49/回撤-47.7%, 完全移除 -78.4%/夏普-1.01/-85.7%。
    2026-09-08 起报告页支持一键切换开/关, 默认开启。
    """
    index_file = index_file or f"{BASE}/data/meta/index_daily.parquet"
    out_file = out_file or f"{BASE}/data/meta/market_regime.parquet"
    idx = pd.read_parquet(index_file).sort_values("date").reset_index(drop=True)
    c = idx["close"]
    ma5, ma10, ma20 = c.rolling(5).mean(), c.rolling(10).mean(), c.rolling(20).mean()
    ma5p, ma10p = ma5.shift(1), ma10.shift(1)
    ret = c.pct_change()
    states = []
    prev = "Z1"
    for i in range(len(idx)):
        if np.isnan(ma20.iloc[i]) or np.isnan(ma5p.iloc[i]):
            states.append("Z1")
            continue
        # S0/S0-2 逃顶空仓
        if (ma5.iloc[i] < ma10.iloc[i] and ma5.iloc[i] < ma20.iloc[i]) or \
           (ma5.iloc[i] < ma20.iloc[i] and ret.iloc[i] < -0.01):
            cur = "Z0"
        # B3 重仓: 均线多头排列
        elif c.iloc[i] > ma5.iloc[i] > ma10.iloc[i] > ma20.iloc[i]:
            cur = "Z3"
        # B2 半仓: 有效站上20日线
        elif c.iloc[i] > ma20.iloc[i]:
            cur = "Z2"
        # B1/B1-1 抄底回补: 5MA/10MA 同时拐头向上
        elif ma5.iloc[i] > ma5p.iloc[i] and ma10.iloc[i] > ma10p.iloc[i]:
            cur = "Z1"
        else:
            # S2 重仓破10日线 -> 半仓; S1 破20日线 -> 轻仓; 其余维持
            if prev == "Z3" and c.iloc[i] < ma10.iloc[i]:
                cur = "Z2"
            elif prev in ("Z2", "Z3") and c.iloc[i] < ma20.iloc[i]:
                cur = "Z1"
            else:
                cur = prev
        states.append(cur)
        prev = cur
    out = pd.DataFrame({"date": pd.to_datetime(idx["date"]), "state": states})
    out["target_ratio"] = out["state"].map({"Z0": 0.0, "Z1": 0.3, "Z2": 0.5, "Z3": 1.0})
    out.to_parquet(out_file, index=False)
    dist = out["state"].value_counts().to_dict()
    print(f"大盘状态机({out_file.split('/')[-1]}): {dist}  "
          f"(最新 {out['date'].iloc[-1]:%Y-%m-%d} = {out['state'].iloc[-1]})", flush=True)
    return out


def signal_momentum(df):
    m1 = df["ret20"].groupby(level=1).rank(pct=True) >= 0.95    # 横截面Top5%
    m2 = (df["close"] > df["ma20"]) & (df["ma20"] > df["ma60"]) & df["ma60_rising"]
    m3 = df["ret20"] < 0.60                                     # 排除极端妖股
    m4 = df["ret120"] > 0
    m5 = df["amt20"] >= 3e7
    sig = m1 & m2 & m3 & m4 & m5
    score = df["ret20"]
    return sig, score


def exit_flags(df):
    """放量滞涨标志 (趋势破位规则已于 2026-09-07 移除: A/B 证实纯负贡献)"""
    shrink = (df["volume"] >= 2 * df["prev_vol_ma5"]) & (df["ret"] < 0) & (df["close"] < df["open"])
    return shrink


def run_scan():
    print("加载K线分片...", flush=True)
    hfq = load_klines("hfq")
    raw = load_klines("raw")
    basic = pd.read_parquet(f"{BASE}/data/meta/stock_basic.parquet")
    # 过滤: ST/退市整理/低价股/北交所(baostock无北交, 天然不含)
    basic = basic[~basic["name"].str.contains("ST|退", na=False)]
    keep = set(basic["secid"])
    hfq = hfq[hfq["code"].isin(keep)]
    raw = raw[raw["code"].isin(keep)]
    print(f"K线加载: hfq {len(hfq):,} 行 / raw {len(raw):,} 行 / 有效股票 {hfq['code'].nunique()}", flush=True)

    df = build_indicators(hfq, raw)
    # 回测窗口: 2023-09-01 起 (指标已在 hfq 全历史上计算, 2023-01 起的前置数据
    # 保证 ret120 在窗口首日即可用 —— 消除"前120日空仓死区"假象, 缺陷③修复)
    df = df[df.index.get_level_values(1) >= pd.Timestamp(START)]
    print("指标计算完成", flush=True)

    market_regime()
    sig_c, score_c = signal_momentum(df)
    shrink = exit_flags(df)

    names = basic.set_index("secid")["name"]

    def collect(sig, score, strat):
        sub = df[sig.fillna(False)]
        out = pd.DataFrame({
            "code": sub.index.get_level_values(0),
            "date": sub.index.get_level_values(1),
            "strategy": strat,
            "score": score[sig.fillna(False)].values,
        })
        return out

    signals = collect(sig_c, score_c, "momentum")
    signals["name"] = signals["code"].map(names)
    # 评分归一: 策略内按日排名分位 (0-1)
    signals["score"] = signals.groupby(["date", "strategy"])["score"].rank(pct=True)
    signals = signals.sort_values(["date", "strategy", "score"], ascending=[True, True, False])
    signals.to_parquet(f"{BASE}/data/meta/signals.parquet", index=False)
    print(f"信号总数: {len(signals):,}  (动量轮动)", flush=True)

    # 退出标志 -> 宽表
    idx = df.index
    flags = pd.DataFrame({
        "shrink": shrink.fillna(False).values,
        "close": df["close"].values,
        "raw_close": df["raw_close"].values,
        "open": df["open"].values,
        "raw_open": df["raw_open"].values,
    }, index=idx)
    flags.reset_index().to_parquet(f"{BASE}/data/meta/flags_long.parquet", index=False)
    return signals


if __name__ == "__main__":
    run_scan()
