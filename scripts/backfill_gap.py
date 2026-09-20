#!/usr/bin/env python3
"""交易日断档回补 —— 补 `data/kline/incremental/` 的整段缺口 (raw + hfq) 并同步三个指数。

## 为什么需要它
`daily_update.update_kline()` 一次只能补**一天**(它按 target_day 抓当日快照), 且
`verify_store.py` 原实现只比 `max(date)` —— 主分片与指数**同时**冻结时会被判"对齐" PASS。
2026-09-13 删掉 qfq 家族但校验清单未同步 -> 流水线卡在最后一步停摆 5 天 ->
09-14~09-18 五个交易日整段缺失, 而每天 cron 都"跑完了"(数据更新步骤发现
`store.max >= target_day` 直接跳过)。这类洞只能按**真实交易日历**定缺口后整段补。

## 用法
    python scripts/backfill_gap.py                 # 自动探测缺口并回补
    python scripts/backfill_gap.py --dry-run       # 只打印缺口与计划, 不写盘
    python scripts/backfill_gap.py --start 2026-09-14 --end 2026-09-18

## 口径 (必须与 backfill_baostock.py / 数据仓一致, 违反即单位混库)
    raw: code,date,open,high,low,close,volume,amount   volume 单位=**股** (baostock 本就是股)
    hfq: code,date,open,high,low,close                 adjustflag=1, 历史值永久冻结
    **绝不写 qfq** (2026-09-13 已整族移除, 信号链路唯一口径 hfq)
"""
import argparse
import datetime
import glob
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META, KDIR = f"{BASE}/data/meta", f"{BASE}/data/kline"
INC = f"{KDIR}/incremental"
FIELDS_RAW = "date,open,high,low,close,volume,amount"
FIELDS_Q = "date,open,high,low,close"
START_FLOOR = "2023-09-01"          # 与 backfill_baostock.py / 数据仓窗口一致
INDICES = {"sh.000001": "index_daily", "sh.000300": "bench_daily", "sh.000852": "csi1000_daily"}


def _query(bs_mod, code: str, adjust: str, start: str, end: str) -> pd.DataFrame | None:
    flds = FIELDS_RAW if adjust == "3" else FIELDS_Q
    rs = bs_mod.query_history_k_data_plus(code, flds, start_date=start, end_date=end,
                                          frequency="d", adjustflag=adjust)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=flds.split(","))
    for c in flds.split(",")[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def trading_calendar(bs_mod, start: str, end: str) -> list[pd.Timestamp]:
    df = _query(bs_mod, "sh.000001", "3", start, end)
    if df is None:
        raise RuntimeError(f"交易日历获取失败 ({start}~{end}) —— 不猜缺口, 直接中止")
    return sorted(pd.to_datetime(df["date"]))


def stored_last_day() -> pd.Timestamp:
    """库内 raw 最新日期 (主分片 + 增量一起看, 增量已是库的一部分)。"""
    files = sorted(glob.glob(f"{KDIR}/raw_*.parquet")) + sorted(glob.glob(f"{INC}/raw_*.parquet"))
    if not files:
        raise RuntimeError(f"找不到任何 raw 分片: {KDIR} —— 仓库根目录不对?")
    return max(pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).max() for f in files)


def listed_codes() -> list[str]:
    """待回补标的: 在市 + ST/退市过滤 —— 与 backfill_baostock.py / signals 一致。"""
    basic = pd.read_parquet(f"{META}/stock_basic.parquet")
    basic = basic[~basic["name"].str.contains("ST|退", na=False)]
    basic = basic[(basic["status"] == "1") | (basic["out_date"] >= pd.Timestamp(START_FLOOR))]
    return sorted(basic["code"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import baostock as bs

    last = stored_last_day()
    today = pd.Timestamp(datetime.date.today())
    bs.login()
    try:
        # 日历窗口从数据仓 floor 起, 不能只从 last-10天 起 —— 指数可能缺**中间某天**:
        # 实测 index_daily 缺 09-08 而它的 max 已是 09-11, 窄窗/只看 max 都看不见。
        cal = trading_calendar(bs, START_FLOOR, today.strftime("%Y-%m-%d"))
        win = lambda d: ((not args.start or d >= pd.Timestamp(args.start))  # noqa: E731
                         and (not args.end or d <= pd.Timestamp(args.end)))
        if args.start or args.end:
            gap = [d for d in cal if win(d)]
        else:
            # 缺口 = 库内最新日之后的交易日, 且 incremental 里还没有对应文件
            gap = [d for d in cal if d > last
                   and not os.path.exists(f"{INC}/raw_{d:%Y%m%d}.parquet")]
        # 指数缺口 = 日历里**缺哪些天**(不是只取 >max): 内部空洞(09-08)只追加末尾永远补不到
        idx_miss = {}
        for name in INDICES.values():
            have = set(pd.to_datetime(pd.read_parquet(f"{META}/{name}.parquet")["date"]))
            idx_miss[name] = [d for d in cal if d not in have and win(d)]
        print(f"库内最新: {last:%Y-%m-%d} | 交易日历最新: {cal[-1]:%Y-%m-%d} | 缺口 {len(gap)} 天: "
              + (", ".join(f"{d:%Y-%m-%d}" for d in gap) if gap else "无"), flush=True)
        for name, miss in idx_miss.items():
            print(f"  {name} 缺 {len(miss)} 天"
                  + (f": {', '.join(f'{d:%Y-%m-%d}' for d in miss[:8])}" if miss else ""), flush=True)
        if not gap and not any(idx_miss.values()):
            print("无缺口, 退出(未写任何文件)", flush=True)
            return
        if args.dry_run:
            print("[dry-run] 不写盘", flush=True)
            return

        codes = listed_codes() if gap else []
        buf_raw: dict[pd.Timestamp, list[pd.DataFrame]] = {d: [] for d in gap}
        buf_hfq: dict[pd.Timestamp, list[pd.DataFrame]] = {d: [] for d in gap}
        n_ok = n_fail = 0
        # ★ K线无缺口 ≠ 无事可做: 指数可能只是有内部空洞(09-08), 此时必须跳过抓取循环,
        #   否则 gap[0] 直接 IndexError (回归测试场景3)。
        if not gap:
            print("K线无缺口, 跳过抓取 (只补指数)", flush=True)
        else:
            print(f"标的 {len(codes)} 只, 逐只取缺口日 → {INC}/...", flush=True)
            for k, code in enumerate(codes):
                # 拉窄窗即可: baostock 后复权值不随查询起点变化 (已实测全量/窄窗逐值一致)
                a = (gap[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                b = gap[-1].strftime("%Y-%m-%d")
                raw = _query(bs, code, "3", a, b)
                hfq = _query(bs, code, "1", a, b)
                if raw is None or hfq is None:
                    n_fail += 1
                    continue
                n_ok += 1
                for c, src, buf in (("raw", raw, buf_raw), ("hfq", hfq, buf_hfq)):
                    cur = src[src["date"].isin([f"{d:%Y-%m-%d}" for d in gap])].copy()
                    if not cur.empty:
                        cur.insert(0, "code", code)
                        for d in gap:
                            want = f"{d:%Y-%m-%d}"
                            row = cur[cur["date"] == want]
                            if not row.empty:
                                buf[d].append(row)
                if (k + 1) % 200 == 0:
                    print(f"  {k + 1}/{len(codes)} ok={n_ok} fail={n_fail}", flush=True)
            print(f"抓取完成: ok={n_ok} fail={n_fail}", flush=True)

        os.makedirs(INC, exist_ok=True)
        for d in gap:
            r = pd.concat(buf_raw[d], ignore_index=True) if buf_raw[d] else pd.DataFrame()
            h = pd.concat(buf_hfq[d], ignore_index=True) if buf_hfq[d] else pd.DataFrame()
            if r.empty:
                print(f"  [WARN] {d:%Y-%m-%d} 无数据, 跳过", flush=True)
                continue
            r = r[["code", "date", "open", "high", "low", "close", "volume", "amount"]]
            h = h[["code", "date", "open", "high", "low", "close"]]
            r.to_parquet(f"{INC}/raw_{d:%Y%m%d}.parquet", index=False)
            h.to_parquet(f"{INC}/hfq_{d:%Y%m%d}.parquet", index=False)
            print(f"  写入 {d:%Y-%m-%d}: raw {len(r)} 行 / hfq {len(h)} 行", flush=True)

        # ---- 指数同步 (否则 K线领先指数 -> verify_store WARN, 且状态机/基准用旧数据) ----
        # 用上面算好的 idx_miss (日历里真实缺的天), 而不是 `d > cur.date.max()` ——
        # 09-08 是**内部**空洞 (max 已是 09-11), 只看 max 永远补不到它。
        for code, name in INDICES.items():
            path = f"{META}/{name}.parquet"
            cur = pd.read_parquet(path)
            cur["date"] = pd.to_datetime(cur["date"])
            miss = idx_miss[name]
            if not miss:
                print(f"  {name} 已是最新 ({cur['date'].max():%Y-%m-%d})", flush=True)
                continue
            add = _query(bs, code, "3", miss[0].strftime("%Y-%m-%d"), miss[-1].strftime("%Y-%m-%d"))
            if add is None:
                print(f"  [WARN] {name} 指数无数据", flush=True)
                continue
            add["date"] = pd.to_datetime(add["date"])
            add = add[~add["date"].isin(cur["date"])]
            add = add[add["date"].isin(miss)]
            if add.empty:
                print(f"  [WARN] {name} 缺 {len(miss)} 天但数据源没返回 (前5: "
                      + ", ".join(f"{d:%Y-%m-%d}" for d in miss[:5]) + ")", flush=True)
                continue
            add = add.reindex(columns=cur.columns)
            pd.concat([cur, add], ignore_index=True).sort_values("date").to_parquet(path, index=False)
            print(f"  {name} 补 {len(add)} 行 ({miss[0]:%Y-%m-%d}~{miss[-1]:%Y-%m-%d}) "
                  f"-> 最新 {max(add['date'].max(), cur['date'].max()):%Y-%m-%d}", flush=True)
            if len(add) < len(miss):
                print(f"    [WARN] 仍缺 {len(miss) - len(add)} 天, 需人工查数据源", flush=True)
    finally:
        bs.logout()

    print("\n回补完成。接着跑 verify_store.py 与 run_daily.py 重建信号/报告。", flush=True)
    print("注意: 缺口整段补完后, 次日 daily_update 会恢复正常增量 (store.max 已推进)。", flush=True)


if __name__ == "__main__":
    sys.exit(main())