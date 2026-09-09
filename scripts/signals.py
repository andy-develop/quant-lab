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
ST 过滤: 2026-09-08 起用 baostock 逐日 isST 历史(st_history.parquet)按信号日
状态过滤, 消除"当前名称过滤"的未来函数。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
START = "2023-09-01"


def load_klines(kind: str) -> pd.DataFrame:
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


def groll(s: pd.Series, w: int, fn: str) -> pd.Series:
    return s.groupby(level=0, sort=False).transform(lambda x: getattr(x.rolling(w, min_periods=w), fn)())


def gshift(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level=0, sort=False).transform(lambda x: x.shift(n))


def build_indicators(hfq: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
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

    # ---- 评分因子 (2026-09-08 引入, 仅影响同日候选的买入优先级, 不改选股条件) ----
    df["vol20"] = groll(df["ret"], 20, "std")                       # 20日日收益波动率
    # 20日夏普 —— 全项目唯一 sharpe 定义 (此前 signal_scores 内有第二套 ret20/vol20 写法, 秩等价但易混淆);
    # vol20=0(20日收益恒定, 极罕见) -> NaN, 评分/横截面排名自然剔除
    df["sharpe20"] = (df["ret20"] / df["vol20"].where(df["vol20"] > 0)) / np.sqrt(20)
    df["win60"] = groll((df["ret"] > 0).astype(float), 60, "mean")  # 60日上涨天数占比
    # 趋势一致性: MA5>MA10>MA20>MA60 的连续天数 (bool 段内计数, 断点归零)
    bull = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    seg = (~bull).groupby(level=0, sort=False).cumsum()
    df["trend_streak"] = bull.groupby([df.index.get_level_values(0), seg], sort=False).cumcount() + 1
    df.loc[~bull, "trend_streak"] = 0
    # Amihud 非流动性: 20日平均 |日收益|/成交额(元); amt=0(停牌) -> NaN, 评分日候选自然剔除
    df["illiq20"] = groll((df["ret"].abs() / df["amt"].replace(0, np.nan)), 20, "mean")
    # KDJ(9,3,3) 的 J 值: RSV -> K/D 递推(ewm alpha=1/3, 按股分组不跨股泄漏) -> J=3K-2D
    hh9 = groll(df["high"], 9, "max")
    ll9 = groll(df["low"], 9, "min")
    rsv = (df["close"] - ll9) / (hh9 - ll9).replace(0, np.nan) * 100
    k = rsv.groupby(level=0, sort=False).transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())
    d = k.groupby(level=0, sort=False).transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())
    df["kdj_j"] = 3 * k - 2 * d
    # J 均值偏离: 20日均J - 当日J。正=昨日J自高位回落(回调), 负=J加速上冲(超买)。
    # "昨日"= 信号日T收盘(对T+1开盘买入而言T日即昨日); 如需 T-1 口径改用 gshift(df["kdj_j"], 1)
    df["j_dev"] = groll(df["kdj_j"], 20, "mean") - df["kdj_j"]

    return df


def market_regime(index_file: str | None = None, out_file: str | None = None) -> pd.DataFrame:
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
    states: list[str] = []
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


def signal_momentum(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    m1 = df["ret20"].groupby(level=1).rank(pct=True) >= 0.95    # 横截面Top5%
    m2 = (df["close"] > df["ma20"]) & (df["ma20"] > df["ma60"]) & df["ma60_rising"]
    m3 = df["ret20"] < 0.60                                     # 排除极端妖股
    m4 = df["ret120"] > 0
    m5 = df["amt20"] >= 3e7
    sig = m1 & m2 & m3 & m4 & m5
    score = df["ret20"]
    return sig, score


SCORE_MODE = "quality_kdj5"    # momentum | quality | quality_kdj5 | quality_kdj10 | amihud | voladj | kdj | kdj_rev —— 2026-09-08 A/B 台账: quality 胜出纯动量/kdj单因子; 晚间混合 A/B: quality_kdj5(开+115.3%/-13.1%/夏普1.45, 全部子区间跑赢基线 quality 开+31.8%) 切默认, kdj10(开+91.6%/关+94.2%) 备选; 回滚改此一行即可
KDJ_W: float | None = None     # A/B 扫描用: 非 None 时 quality_kdj5 的 kdj 权重取该值(从低波动权重 0.10 匀出, 其余四因子不动); None=生产口径 kdj=0.05


def signal_scores(df: pd.DataFrame, idx_vol: pd.Series | None = None) -> dict[str, pd.Series]:
    """候选股买入优先级评分 (0-1, 按日横截面排名分位)。只影响同日多信号时的
    买入顺序/名额竞争, 不改变选股条件。四套口径 A/B 后择优:
      momentum: 纯 20 日动量 (原版)
      quality : 中泰金工趋势质量合成 0.3动量+0.25夏普+0.2胜率+0.15趋势一致性+0.1低波动
      amihud  : 动量 × 流动性分位 (Amihud 非流动性倒数排名; 2026-09-08 淘汰)
      voladj  : 调整后动量 = ret20/vol20 × 当日指数20日波动率。注: 引擎按日横截面
                排序消费 score, 指数波动率是日级常数, 乘上去秩不变 —— 实际生效部分
                是 ret20/vol20 (波动率调整动量/动量夏普); 保留因子是为公式忠实。
      kdj     : KDJ均值偏离 = 20日均J - 信号日J (用户公式)。值大=J自高位回落,
                偏好强势股中的短期回调 (低吸方向)
      kdj_rev : kdj 的反向对照 (-j_dev), 偏好 J 加速上冲 (追涨方向), 用于检验方向
      quality_kdj5 / quality_kdj10 : kdj 以 5%/10% 微权重并入 quality 合成 (A/B 用)
      rrf     : 等权 Reciprocal Rank Fusion —— 六因子各自单独排整数秩, 得分=Σ 1/(5+rank)
                (2026-09-09 A/B 灾难性淘汰, 保留作对照)
      wrrf    : 加权 RRF —— Σ weight_i/(2+rank_i), 权重同 quality_kdj5, k=2 (A/B 用)
    """
    rk = lambda s: s.groupby(level=1).rank(pct=True)
    mom = rk(df["ret20"])
    voladj = rk(df["sharpe20"])   # 调整后动量: 秩上 ≡ ret20/vol20 (sqrt(20) 是年化常数不改排序);
                                  # ×当日指数20日波动率亦不改同日横截面排序, 保留乘法为公式忠实
    if idx_vol is not None:
        voladj = voladj * pd.Series(df.index.get_level_values(1).map(idx_vol).to_numpy(), index=df.index)
    quality = (0.30 * mom
               + 0.25 * rk(df["sharpe20"])
               + 0.20 * rk(df["win60"])
               + 0.15 * rk(df["trend_streak"])
               + 0.10 * rk(-df["vol20"]))
    # quality + kdj 混合: kdj(20日均J-信号日J, 低吸方向)以微权重并入 quality 合成。
    # kdj5: 从最低权重因子(-vol 0.10)匀 0.05 给 kdj; kdj10: 五因子等比缩放×0.9 腾出 0.10。
    # KDJ_W 非 None 时为权重敏感性扫描模式: kdj 取 KDJ_W, 低波动取 0.10-KDJ_W, 其余四因子不动。
    kw = 0.05 if KDJ_W is None else float(KDJ_W)
    kdj = rk(df["j_dev"])
    quality_kdj5 = (0.30 * mom + 0.25 * rk(df["sharpe20"]) + 0.20 * rk(df["win60"])
                    + 0.15 * rk(df["trend_streak"]) + (0.10 - kw) * rk(-df["vol20"]) + kw * kdj)
    quality_kdj10 = (0.27 * mom + 0.225 * rk(df["sharpe20"]) + 0.18 * rk(df["win60"])
                     + 0.135 * rk(df["trend_streak"]) + 0.09 * rk(-df["vol20"]) + 0.10 * kdj)
    # RRF 家族 (A/B 对照):
    # rrf   : 等权 Reciprocal Rank Fusion —— 六因子各自单独排整数秩(秩1=最优), Σ 1/(5+rank)。
    #         2026-09-09 A/B 灾难性淘汰(开 -26.7% vs 基线 +114.1%): 等权抹掉动量主导权重
    #         + 倒数折扣压缩头部差距(引擎只买前3) → 动量稀释成平庸均衡。
    # wrrf  : 加权 RRF —— Σ weight_i/(2+rank_i), 权重与 quality_kdj5 完全一致
    #         (动量0.30/夏普0.25/胜率0.20/趋势0.15/低波动0.05/kdj0.05), k=2 减缓头部压缩。
    #         与 quality_kdj5 的唯一差异: 百分比秩(线性) → 倒数秩(非线性折扣, 尾部仍占小权重)。
    rint = lambda s, asc=True: s.groupby(level=1).rank(method="average", ascending=asc)
    rrf = (1.0 / (5 + rint(df["ret20"])) + 1.0 / (5 + rint(df["sharpe20"]))
           + 1.0 / (5 + rint(df["win60"])) + 1.0 / (5 + rint(df["trend_streak"]))
           + 1.0 / (5 + rint(df["vol20"], asc=False)) + 1.0 / (5 + rint(df["j_dev"])))
    wrrf = (0.30 / (2 + rint(df["ret20"])) + 0.25 / (2 + rint(df["sharpe20"]))
            + 0.20 / (2 + rint(df["win60"])) + 0.15 / (2 + rint(df["trend_streak"]))
            + 0.05 / (2 + rint(df["vol20"], asc=False)) + 0.05 / (2 + rint(df["j_dev"])))
    amihud = mom * rk(1.0 / df["illiq20"])
    return {"momentum": mom, "quality": quality, "quality_kdj5": quality_kdj5,
            "quality_kdj10": quality_kdj10, "rrf": rrf, "wrrf": wrrf,
            "amihud": amihud, "voladj": voladj,
            "kdj": kdj, "kdj_rev": rk(-df["j_dev"])}


def exit_flags(df: pd.DataFrame) -> pd.Series:
    """放量滞涨标志 (趋势破位规则已于 2026-09-07 移除: A/B 证实纯负贡献)"""
    shrink = (df["volume"] >= 2 * df["prev_vol_ma5"]) & (df["ret"] < 0) & (df["close"] < df["open"])
    return shrink


def run_scan(score_mode: str | None = None, out_file: str | None = None) -> pd.DataFrame:
    """score_mode/out_file: A/B 用 —— 指定本次扫描的评分模式与信号输出路径;
    默认 None 走生产口径 (SCORE_MODE / data/meta/signals.parquet), 行为不变。"""
    mode = score_mode or SCORE_MODE
    out_file = out_file or f"{BASE}/data/meta/signals.parquet"
    t0 = time.time()
    print("加载K线分片...", flush=True)
    hfq = load_klines("hfq")
    raw = load_klines("raw")
    basic = pd.read_parquet(f"{BASE}/data/meta/stock_basic.parquet")
    st = pd.read_parquet(f"{BASE}/data/meta/st_history.parquet")
    st_last = pd.to_datetime(st["date"]).max()
    lag = (pd.Timestamp.today().normalize() - st_last).days
    if lag > 20:
        # 静默失败链防护: baostock 连续数周不可用时, 新戴帽股会持续漏过滤且回测失真无人察觉
        raise SystemExit(f"FATAL: st_history 已 {lag} 天未更新(最新 {st_last:%Y-%m-%d}) —— "
                         f"新戴帽个股漏过滤风险不可接受, 先跑 backfill_st.py incw/incmerge 刷新再扫描")
    if lag > 10:
        print(f"WARN: st_history 已 {lag} 天未更新(最新 {st_last:%Y-%m-%d}), "
              f"尽快跑 backfill_st.py inc 刷新", flush=True)
    # ST/退市 过滤 (2026-09-08 晚升级):
    # 1) ST: 逐日 isST 状态 (st_history.parquet), 消除"当前名称过滤"的未来函数
    # 2) 退市: out_date 已知 -> 精确到行, 仅剔 [out_date-30天, ∞) —— 健康期信号保留, 回测更真实;
    #    无 out_date 但名称含"退"(退市整理期未摘牌/字段缺失) -> 兜底整段剔除 (保守方向)
    if "out_date" in basic.columns:
        _cut = pd.to_datetime(basic["out_date"], errors="coerce") - pd.Timedelta(days=30)
        cut_map = pd.Series(_cut.to_numpy(), index=basic["secid"]).dropna()
    else:
        cut_map = pd.Series(dtype="datetime64[ns]")
    retire = set(basic.loc[~basic["secid"].isin(cut_map.index)
                           & basic["name"].str.contains("退", na=False), "secid"])
    print(f"K线加载: hfq {len(hfq):,} 行 / raw {len(raw):,} 行 / 股票 {hfq['code'].nunique()}", flush=True)

    df = build_indicators(hfq, raw)
    # 回测窗口: 2023-09-01 起 (指标已在 hfq 全历史上计算, 2023-01 起的前置数据
    # 保证 ret120 在窗口首日即可用 —— 消除"前120日空仓死区"假象, 缺陷③修复)
    df = df[df.index.get_level_values(1) >= pd.Timestamp(START)]
    # 逐日 ST 剔除: merge isST==1 的 (code,date) 集合
    idx_df = df.index.to_frame(index=False)
    idx_df = idx_df.merge(st.loc[st["is_st"] == 1, ["code", "date"]].assign(_bad=True),
                          on=["code", "date"], how="left")
    bad = idx_df["_bad"].fillna(False).astype(bool).to_numpy()
    n_st_rows = int(bad.sum())
    # 退市行剔除: (code,date) 中 date >= cut(out_date-30天) 的行, 未匹配到 cut_map 的为 NaT(保留)
    _dates = df.index.get_level_values(1).to_numpy()
    _cut_for = pd.Series(df.index.get_level_values(0)).map(cut_map).to_numpy()
    bad_retire = pd.notna(_cut_for) & (_dates >= _cut_for)
    n_retire_rows = int(bad_retire.sum())
    in_retire = df.index.get_level_values(0).isin(retire)
    df = df[~bad & ~bad_retire & ~in_retire]
    print(f"ST 过滤: 剔除 ST日行 {n_st_rows:,} / 退市行 {n_retire_rows:,}(out_date精确) / "
          f"整段剔除股 {len(retire)} 只(无out_date兜底), 剩余 {len(df):,} 行 / "
          f"{df.index.get_level_values(0).nunique()} 只", flush=True)
    print("指标计算完成", flush=True)
    print(f"[计时] 加载+指标+过滤 {time.time()-t0:.0f}s", flush=True)

    market_regime()
    idx_close = pd.read_parquet(f"{BASE}/data/meta/index_daily.parquet").set_index("date")["close"].sort_index()
    idx_vol20 = idx_close.pct_change().rolling(20).std()
    sig_c, _ = signal_momentum(df)
    score_c = signal_scores(df, idx_vol=idx_vol20)[mode]
    shrink = exit_flags(df)

    names = basic.set_index("secid")["name"]

    def collect(sig: pd.Series, score: pd.Series, strat: str) -> pd.DataFrame:
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
    # 评分归一: 策略内按日排名分位 (0-1); 合成评分本身已秩归一, 再排名不改变顺序
    signals["score"] = signals.groupby(["date", "strategy"])["score"].rank(pct=True)
    signals = signals.sort_values(["date", "strategy", "score"], ascending=[True, True, False])
    signals.to_parquet(out_file, index=False)
    print(f"信号总数: {len(signals):,}  (动量轮动, 评分模式={mode}, 输出={out_file.split('/')[-1]})", flush=True)

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
