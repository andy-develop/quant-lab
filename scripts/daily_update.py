#!/usr/bin/env python3
"""每日收盘后增量更新:
1. 全市场快照(腾讯批量, ~93请求) -> 当日K线追加(不复权raw含真实成交额 + hfq按冻结因子折算)
2. 除权检测: 快照昨收 != 库内最后收盘 -> 整段重拉该股(raw直接修复, hfq由新qfq派生)
3. 涨停池/炸板池(近10个交易日缺失的补齐)
4. 输出汇总 data/pool/summary.csv
用法: python daily_update.py
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
META, KDIR, POOL = f"{BASE}/data/meta", f"{BASE}/data/kline", f"{BASE}/data/pool"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ★ §0.4 修复：快照解析（含市场判定）抽到 quant-hub/common/datasource/vendor_tencent.py，
#   本文件不再自行 split/startswith 判市场。合并过渡期若 quant-hub 未就位，
#   回落到同目录 vendored 副本，保证 quant-lab 单独可跑。
try:
    from common.datasource.vendor_tencent import parse_qt_batch_response
    from common.datasource import FetchStats
    from common.gates.coverage import check_stock_coverage
except ImportError:  # pragma: no cover - 合并过渡期
    from _vendor_tencent import parse_qt_batch_response  # type: ignore
    from _vendor_tencent import FetchStats, check_stock_coverage  # type: ignore


def make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update(UA)
    return s


def load_store() -> pd.DataFrame:
    shards = sorted(glob.glob(f"{KDIR}/raw_*.parquet"))
    incs = sorted(glob.glob(f"{KDIR}/incremental/raw_*.parquet"))
    fixs = sorted(glob.glob(f"{KDIR}/fixup/raw_*.parquet"))  # 只取 raw 修复件, glob 到 hfq_*.parquet 会把复权行拼进不复权库
    dfs = [pd.read_parquet(f) for f in shards + incs]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # 统一代码格式为腾讯式 (1.600000/0.000001): 历史分片是 baostock 式 (sh./sz.),
    # 快照增量是腾讯式 —— 不统一会导致 ratio/除权检测的键全部 miss (冷启动事故)。
    # 必须在 fixup 覆盖之前统一: fixup 文件名/内容是腾讯式, 历史分片是 baostock 式,
    # 顺序颠倒会让覆盖 isin 全 miss -> 整段重复行 (2026-09-08 verify_store 发现)
    df["code"] = df["code"].str.replace("sh.", "1.", regex=False).str.replace("sz.", "0.", regex=False)
    if fixs:
        fix = pd.concat([pd.read_parquet(f) for f in fixs], ignore_index=True)
        df = df[~df["code"].isin(set(fix["code"]))]
        df = pd.concat([df, fix], ignore_index=True)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


BATCH_SIZE = 60          # 与旧实现一致（实测 ~93 请求覆盖全 A）
BATCH_RETRY = 3          # 批次整体失败的重试次数（指数退避 1s/2s/4s）


def _sym_map(secids: list[str]) -> dict[str, str]:
    """secid(1.600004) -> 腾讯符号(sh600004)，作为解析器的反查表。"""
    out: dict[str, str] = {}
    for secid in secids:
        mkt, num = str(secid).split(".")
        out[("sh" if mkt == "1" else "sz") + num] = secid
    return out


def _fetch_batch(s: requests.Session, batch: list[str], stats,
                 smap: dict[str, str]) -> list[dict]:
    """取一批快照。

    旧实现批次异常只 `sleep(2); continue` —— 沪市被限流时整 38 个批次全丢且无痕。
    新实现：批次指数退避重试 3 次 → 仍失败则逐只再试一次 → 每次失败都记账。
    """
    last_exc: Exception | None = None
    for attempt in range(BATCH_RETRY):
        try:
            r = s.get("https://qt.gtimg.cn/q=" + ",".join(batch), timeout=15)
            r.encoding = "gbk"
            return parse_qt_batch_response(r.text, smap, stats)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            stats.dropped_batch_exc += 1
            time.sleep(2 ** attempt)
    # 重试耗尽：逐只再试一次（沪市限流常是整批被拒，单只请求存活率更高）
    rows: list[dict] = []
    for sym in batch:
        try:
            r = s.get("https://qt.gtimg.cn/q=" + sym, timeout=15)
            r.encoding = "gbk"
            rows += parse_qt_batch_response(r.text, smap, stats)
        except Exception:  # noqa: BLE001
            stats.dropped_batch_exc += 1
        time.sleep(0.15)
    if not rows:
        print(f"[WARN] 批次重试后仍为空({len(batch)}只): {last_exc}", flush=True)
    return rows


def snapshot_day(s: requests.Session, secids: list[str], day_ts: pd.Timestamp) -> pd.DataFrame:
    """腾讯批量快照 -> 当日OHLCV (不复权)。

    ★ §0.4 修复版：市场判定交给 parse_qt_batch_response（显式识别 v_s_ 降级前缀），
      四条丢弃路径全部计数并回传，绝不静默。
    """
    smap = _sym_map(secids)
    stats = FetchStats(vendor="tencent-qt")
    syms = list(smap)
    rows: list[dict] = []
    for i in range(0, len(syms), BATCH_SIZE):
        rows += _fetch_batch(s, syms[i:i + BATCH_SIZE], stats, smap)
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = day_ts

    print(f"[snapshot] 取到 {len(df)}/{len(secids)} 只 | 批失败 {stats.dropped_batch_exc} "
          f"短格式 {stats.dropped_short_format} 未知码 {stats.dropped_unknown_code} "
          f"解析失败 {stats.dropped_parse_error}", flush=True)
    # 丢弃统计进 runlog，供覆盖率门禁与事后复盘
    os.makedirs(f"{BASE}/state/data/runlog", exist_ok=True)
    with open(f"{BASE}/state/data/runlog/snapshot_{day_ts:%Y%m%d}.json", "w") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
    snapshot_day.last_stats = stats  # type: ignore[attr-defined]
    return df


def _gate_snapshot(df: pd.DataFrame, basic: pd.DataFrame, day_ts: pd.Timestamp) -> None:
    """覆盖率硬门禁：<95% 黄灯告警，<80% 红灯中止（不写盘）。

    §0.4 的 39/2316=1.7% 沪市覆盖率必须在写盘前拦下 —— 旧实现没有任何断言，
    于是坏数据直接进 incremental/ 并被后续信号消费。
    """
    got = df["code"].tolist() if not df.empty else []
    uni = basic[basic["status"] == "1"][["code"]].copy()
    res = check_stock_coverage(got, uni, day=f"{day_ts:%Y-%m-%d}", raise_on_red=False)
    print("  " + res.summary(), flush=True)
    if res.level == "red":
        raise RuntimeError(
            f"[§0.4 门禁] {day_ts:%Y-%m-%d} 快照覆盖率 {res.overall:.1%} <80%"
            f"（沪 {res.by_market.get('sh', 0):.1%} / 深 {res.by_market.get('sz', 0):.1%}）"
            f" —— 拒绝写盘，等待 21:00 补跑")
    if res.level == "yellow":
        print(f"  [WARN] 覆盖率 {res.overall:.1%} 低于 95%，已标记黄灯", flush=True)


def update_kline() -> pd.Timestamp:
    import datetime
    idx = pd.read_parquet(f"{META}/index_daily.parquet")
    last_idx_day = idx["date"].max()
    # 目标日: 收盘后的交易日取当天, 否则取指数日线的最后一天
    # 统一用北京时间判断 (本地/CI 时钟可能是 UTC)
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(
        datetime.timezone(datetime.timedelta(hours=8)))
    today = pd.Timestamp(now.date())
    if today.weekday() < 5 and now.time() > datetime.time(15, 5) and today > last_idx_day:
        target_day = today
    else:
        target_day = last_idx_day
    basic = pd.read_parquet(f"{META}/stock_basic.parquet")
    listed = basic[basic["status"] == "1"]["secid"].tolist()
    inc_path = f"{KDIR}/incremental/raw_{target_day:%Y%m%d}.parquet"
    if os.path.exists(inc_path):
        print(f"快照已存在: {target_day:%Y-%m-%d}, 跳过K线更新", flush=True)
        return target_day
    store = load_store()
    if store["date"].max() >= target_day:
        print(f"库内已有 {store['date'].max():%Y-%m-%d} 数据, 跳过K线更新", flush=True)
        return target_day
    # 上一交易日以 K线库最后一天为准 (此前用 index_daily 最后一天 —— 指数快照静默
    # 落后时, 快照昨收 vs 库内收盘整体错位, 除权检测假阳性爆发)
    prev_trade_day = store["date"].max()
    prev_rows = store[store["date"] == prev_trade_day].set_index("code")["close"]

    s = make_session()
    # 真实交易日校验 (#18): 节假日(非周末)快照通道无法识别 —— target_day 判定只看
    # "周几+15:05", 国庆等长假会连写 5 天幻影数据(上一交易日行情标成新日期), 且指数/K线
    # 同步幻影, verify_store 也发现不了。以指数日K为准: 当天无 bar 即非交易日, 跳过更新。
    # (已验证: 过去节假日当天查询 ifzq, bars 截至上一交易日且不含当天; 未来日期返回空数组,
    #  空数组按探测失败处理 -> 按原行为继续, 避免网络问题导致漏掉真实交易日)
    try:
        _start = (target_day - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        _r = s.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                   params={"param": f"sh000001,day,{_start},{target_day:%Y-%m-%d},20,"}, timeout=15)
        _bars = [b for b in (_r.json()["data"]["sh000001"].get("day") or [])
                 if isinstance(b, list) and len(b) >= 6]
        if _bars and not any(b[0] == f"{target_day:%Y-%m-%d}" for b in _bars):
            print(f"{target_day:%Y-%m-%d} 非交易日 (节假日, 指数日K无当日bar), 跳过K线更新", flush=True)
            return last_idx_day
    except Exception as e:
        print(f"[WARN] 交易日探测失败(按交易日继续): {e}", flush=True)
    snap = snapshot_day(s, listed, target_day)
    if snap.empty:
        print("快照获取失败!", flush=True)
        return target_day
    # ★ §0.4：覆盖率硬门禁 —— 沪市 1.7% 这类故障必须在写盘前中止
    _gate_snapshot(snap, basic, target_day)
    # 指数快照 -> 追加指数日线 (下个交易日历前提)
    # 双通道: qt.gtimg 快照(重试3次) -> web.ifzq 日K兜底(无 amount, 置0)。
    # 此前仅 WARN 静默落后: 指数落后 -> 状态机/基准用旧数据, 且曾引发
    # prev_trade_day 错位 -> 除权检测假阳性爆发 (2026-09-09 事故, 见台账第10条)
    def _idx_from_snapshot(sym: str) -> dict | None:
        r = None
        for attempt in range(3):
            try:
                r = s.get(f"https://qt.gtimg.cn/q={sym}", timeout=15)
                r.encoding = "gbk"
                line = next((x for x in r.text.split(";") if x.startswith(f"v_{sym}=")), None)
                parts = line.split("=", 1)[1].strip('"').split("~")
                return {"date": target_day, "open": float(parts[5]), "close": float(parts[3]),
                        "high": float(parts[33]), "low": float(parts[34]), "volume": float(parts[6]),
                        "amount": float(parts[37]) * 1e4 if parts[37] else 0.0}
            except Exception as e:
                if attempt == 2:
                    tail = r.text[:80] if r is not None else "no-response"
                    print(f"[WARN] {sym} 快照解析失败(已重试3次): {e}; resp[:80]={tail!r}", flush=True)
                time.sleep(2 + 3 * attempt)
        return None

    def _idx_from_kline(sym: str) -> dict | None:
        try:
            r = s.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                      params={"param": f"{sym},day,2026-08-01,{target_day:%Y-%m-%d},20,"}, timeout=15)
            d = r.json()["data"][sym]
            bars = [b for b in (d.get("day") or []) if isinstance(b, list) and len(b) >= 6]
            if not bars or pd.to_datetime(bars[-1][0]) != target_day:
                return None
            b = bars[-1]
            return {"date": target_day, "open": float(b[1]), "close": float(b[2]),
                    "high": float(b[3]), "low": float(b[4]), "volume": float(b[5]), "amount": 0.0}
        except Exception as e:
            print(f"[WARN] {sym} kline 兜底失败: {e}", flush=True)
            return None

    idx_fail = []
    for sym, p in [("sh000001", f"{META}/index_daily.parquet"),
                   ("sh000300", f"{META}/bench_daily.parquet"),
                   ("sh000852", f"{META}/csi1000_daily.parquet")]:
        row = _idx_from_snapshot(sym) or _idx_from_kline(sym)
        if row is None:
            idx_fail.append(sym)
            continue
        d = pd.read_parquet(p)
        if not (d["date"] == target_day).any():
            d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
            d["date"] = pd.to_datetime(d["date"])
            d.sort_values("date").to_parquet(p, index=False)
    if idx_fail:
        raise RuntimeError(f"指数日线更新失败 {idx_fail} (快照+K线双通道均不可用) —— "
                           f"指数是仓位状态机/基准的输入, 宁可中止也不带病跑 (run_daily 对本步骤 fatal)")
    # 除权检测: 快照prev_close vs 上一交易日库内收盘 (容差0.5%吸收精度差)
    snap["stored_prev"] = snap["code"].map(prev_rows)
    snap["is_div"] = (snap["stored_prev"].notna()) & ((snap["prev_close"] / snap["stored_prev"] - 1).abs() > 0.005)
    div_codes = snap.loc[snap["is_div"], "code"].tolist()
    # 保险丝: 正常交易日除权 ~几十只; 超 30% 必是交易日错位型假阳性,
    # 若继续会整段重拉两千只 -> 数据源限流 -> 空响应 fatal (2026-09-09 事故)
    if len(div_codes) > max(50, 0.3 * len(snap)):
        raise RuntimeError(f"除权检测异常: {len(div_codes)}/{len(snap)} 只被标记 (>30%) —— "
                           f"大概率交易日错位假阳性, 中止以免整段重拉打爆数据源限流")
    print(f"快照 {len(snap)} 只, 检测到除权 {len(div_codes)} 只", flush=True)
    # hfq 折算因子: 库内最后一条 hfq_close / raw_close (后复权因子, 非除权日恒定)
    # hfq = raw × 因子 —— 与 qfq 不同, hfq 历史值永久冻结, 信号可复现 (缺陷④修复)
    last_rows = store.groupby("code").tail(1).set_index("code")[["close"]]
    hfq_all = sorted(glob.glob(f"{KDIR}/hfq_*.parquet")) + sorted(glob.glob(f"{KDIR}/incremental/hfq_*.parquet"))
    hfq_last = pd.concat([pd.read_parquet(f) for f in hfq_all], ignore_index=True)
    hfq_last["date"] = pd.to_datetime(hfq_last["date"])
    # 与 load_store 同口径统一代码格式 (hfq 分片是 baostock 式, 增量是腾讯式)
    hfq_last["code"] = hfq_last["code"].str.replace("sh.", "1.", regex=False).str.replace("sz.", "0.", regex=False)
    hfq_last = hfq_last.groupby("code").tail(1).set_index("code")[["close"]].rename(columns={"close": "hfq_close"})
    rr = last_rows.join(hfq_last, how="inner")
    ratio: dict[str, float] = (rr["hfq_close"] / rr["close"]).replace([np.inf, -np.inf], np.nan).dropna().to_dict()
    # 除权股整段重拉(修复): raw 网络重拉 + 由新 qfq 派生整段 hfq
    os.makedirs(f"{KDIR}/fixup", exist_ok=True)
    for code in div_codes:
        sym = ("sh" if code.startswith("1.") else "sz") + code.split(".")[1]
        fix = refetch_one(s, code, sym)
        if fix is not None:
            fix[0].to_parquet(f"{KDIR}/fixup/raw_{code.replace('.', '_')}.parquet", index=False)
            hfq_fix = derive_hfq_fixup(code, fix[1])
            if hfq_fix is not None:
                hfq_fix.to_parquet(f"{KDIR}/fixup/hfq_{code.replace('.', '_')}.parquet", index=False)
        time.sleep(0.2)
    # 除权股当日行取自修复序列; 其余用折算
    snap_ok = snap[~snap["is_div"]].copy()
    snap_ok["adj"] = snap_ok["code"].map(ratio).fillna(1.0)
    inc_raw = snap_ok[["code", "date", "open", "close", "high", "low", "volume", "amount"]].reset_index(drop=True)
    ohlc_hfq = snap_ok[["open", "close", "high", "low"]].mul(snap_ok["adj"], axis=0).reset_index(drop=True)
    inc_hfq = pd.concat([snap_ok[["code", "date"]].reset_index(drop=True), ohlc_hfq], axis=1)
    os.makedirs(f"{KDIR}/incremental", exist_ok=True)
    inc_raw.to_parquet(inc_path, index=False)
    inc_hfq.to_parquet(f"{KDIR}/incremental/hfq_{target_day:%Y%m%d}.parquet", index=False)
    print(f"K线增量入库: {len(inc_raw)} 只 ({target_day:%Y-%m-%d}), 除权修复 {len(div_codes)} 只", flush=True)
    return target_day


def derive_hfq_fixup(code: str, qfq_fresh: pd.DataFrame) -> pd.DataFrame | None:
    """由整段重拉的新 qfq 派生 hfq: hfq = qfq × K, K = hfq库内最后收盘 / qfq同日收盘。
    (hfq/qfq 对同一快照为常数 —— 两者都是 raw×复权因子, 因子比值不随日期变)
    库内无该股 hfq 时返回 None (K 无法锚定)。"""
    fixs = sorted(glob.glob(f"{KDIR}/hfq_*.parquet") + glob.glob(f"{KDIR}/incremental/hfq_*.parquet")
                  + glob.glob(f"{KDIR}/fixup/hfq_{code.replace('.', '_')}.parquet"))
    if not fixs:
        return None
    dfs = [pd.read_parquet(f) for f in fixs]
    hf = pd.concat(dfs, ignore_index=True)
    hf["date"] = pd.to_datetime(hf["date"])
    hf["code"] = hf["code"].str.replace("sh.", "1.", regex=False).str.replace("sz.", "0.", regex=False)
    hf = hf[hf["code"] == code].sort_values("date")
    if hf.empty:
        return None
    last_date = hf["date"].iloc[-1]
    q = qfq_fresh.copy()
    q["date"] = pd.to_datetime(q["date"])
    anchor = q[q["date"] == last_date]
    if anchor.empty or not hf["close"].iloc[-1]:
        return None
    k = hf["close"].iloc[-1] / anchor["close"].iloc[0]
    out = qfq_fresh[["code", "date", "open", "close", "high", "low"]].copy()
    for c_ in ["open", "close", "high", "low"]:
        out[c_] = out[c_] * k
    return out


def refetch_one(s: requests.Session, code: str, sym: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    def get(fq: str) -> pd.DataFrame | None:
        # 腾讯单次约800根上限, 分两段拼接 (2023-01 起约950个交易日)
        out = []
        for a, b in [("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
            r = s.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                      params={"param": f"{sym},day,{a},{b},800,{fq}"}, timeout=15)
            d = r.json()["data"][sym]
            bars = d.get("qfqday" if fq == "qfq" else "day") or d.get("day")
            if bars:
                out += [bar for bar in bars if isinstance(bar, list) and len(bar) >= 6]
        if not out:
            return None
        df = pd.DataFrame([x[:6] for x in out], columns=["date", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        for c_ in ["open", "close", "high", "low", "volume"]:
            df[c_] = pd.to_numeric(df[c_], errors="coerce")
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df.insert(0, "code", code)
        return df
    raw, qfq = get(""), get("qfq")
    if raw is None or qfq is None:
        return None
    return raw, qfq


def update_pool(last_day: pd.Timestamp | None = None) -> None:
    s = make_session()
    idx = pd.read_parquet(f"{META}/index_daily.parquet")
    recent = idx["date"].tail(10).dt.strftime("%Y%m%d").tolist()
    summary_path = f"{POOL}/summary.csv"
    summary = pd.read_csv(summary_path, dtype={"date": str}) if os.path.exists(summary_path) else pd.DataFrame()
    have = set(summary["date"]) if len(summary) else set()
    for d in recent:
        if d in have:
            continue
        zt = fetch_pool(s, "getTopicZTPool", d)
        zb = fetch_pool(s, "getTopicZBPool", d)
        os.makedirs(POOL, exist_ok=True)
        if zt is not None and len(zt):
            zt.to_parquet(f"{POOL}/zt_{d}.parquet", index=False)
        if zb is not None and len(zb):
            zb.to_parquet(f"{POOL}/zb_{d}.parquet", index=False)
        n_zt = 0 if zt is None else len(zt)
        n_zb = 0 if zb is None else len(zb)
        max_lb = int(zt["lbc"].max()) if zt is not None and len(zt) else 0
        lb_dist = ""
        if zt is not None and len(zt):
            vc = zt["lbc"].value_counts().sort_index()
            lb_dist = json.dumps({int(k): int(v) for k, v in vc.items() if k <= 10}, ensure_ascii=False)
        row = pd.DataFrame([{"date": d, "zt_count": n_zt, "zb_count": n_zb,
                             "max_lb": max_lb, "lb_dist": lb_dist}])
        summary = pd.concat([summary, row], ignore_index=True)
        time.sleep(0.3)
    summary = summary.drop_duplicates("date").sort_values("date")
    summary.to_csv(summary_path, index=False)
    print(f"涨停池汇总: {len(summary)} 个交易日 (最新 {summary['date'].iloc[-1] if len(summary) else '-'})", flush=True)


def fetch_pool(s: requests.Session, fn: str, date_str: str) -> pd.DataFrame | None:
    try:
        r = s.get(f"https://push2ex.eastmoney.com/{fn}",
                  params={"ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt",
                          "Pageindex": "0", "pagesize": "500", "sort": "fbt:asc", "date": date_str},
                  timeout=15)
        data = (r.json() or {}).get("data") or {}
        pool = data.get("pool") or []
        return pd.DataFrame(pool)
    except Exception:
        return None


if __name__ == "__main__":
    day = update_kline()
    update_pool(day)
    print("每日数据更新完成", flush=True)
