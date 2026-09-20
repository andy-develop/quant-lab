#!/usr/bin/env python3
"""quant-lab 底层数据完整性校验 (只读, 不修改任何数据)。
用法: python verify_store.py [仓库根目录]
缺省目录按 /Users/andy/WorkBuddy/*/quant-lab 自动探测。
退出码: 0=通过, 1=存在 FAIL 项。
"""
import glob
import os
import sys

import pandas as pd

FAMILIES = ["raw", "hfq"]  # qfq 已于 2026-09-13 从数据仓移除(信号链路全用 hfq, 见 signals.py)
META_REQUIRED = ["stock_basic", "index_daily", "bench_daily", "csi1000_daily"]


def find_repo():
    if len(sys.argv) > 1:
        return sys.argv[1]
    # CI/本地通用: 脚本自身位于 <repo>/scripts/ 下, 上一级即仓库根
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(here, "data", "kline")):
        return here
    hits = sorted(glob.glob("/Users/andy/WorkBuddy/*/quant-lab"))
    return hits[-1] if hits else None


def load_family(kdir, fam):
    files = sorted(glob.glob(f"{kdir}/{fam}_*.parquet"))
    incs = sorted(glob.glob(f"{kdir}/incremental/{fam}_*.parquet"))
    fixs = sorted(glob.glob(f"{kdir}/fixup/{fam}_*.parquet"))
    if not (files or incs or fixs):
        return None, 0
    dfs = [pd.read_parquet(f) for f in files + incs]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # 必须在 fixup 覆盖之前统一代码格式 (历史分片 baostock 式, fixup 腾讯式),
    # 否则覆盖 isin 全 miss -> 整段重复行 (与 daily_update.load_store 同款坑)
    df["code"] = df["code"].str.replace("sh.", "1.", regex=False).str.replace("sz.", "0.", regex=False)
    if fixs:  # fixup 覆盖同名股票全历史 (与 daily_update.load_store 一致)
        fix = pd.concat([pd.read_parquet(f) for f in fixs], ignore_index=True)
        df = df[~df["code"].isin(set(fix["code"]))]
        df = pd.concat([df, fix], ignore_index=True)
    return df, len(files) + len(incs) + len(fixs)


def trading_days(start, end):
    """真实交易日历 (baostock sh.000001 日K 的日期集合)。取不到返回 None -> 调用方降级 WARN。

    ★ 断档校验不能拿 index_daily 当日历: 指数与主分片会同时冻结(2026-09-14~09-18
      就是两者都停在 09-11 -> 原实现比 max(date) 判"与指数对齐" PASS 而漏掉 5 天空洞)。
    """
    try:
        import baostock as bs
    except ImportError:
        print("[WARN] baostock 未安装, 跳过交易日断档校验")
        return None
    try:
        bs.login()
        rs = bs.query_history_k_data_plus("sh.000001", "date", start_date=f"{start:%Y-%m-%d}",
                                          end_date=f"{end:%Y-%m-%d}", frequency="d", adjustflag="3")
        out = []
        while rs.error_code == "0" and rs.next():
            out.append(pd.Timestamp(rs.get_row_data()[0]))
        bs.logout()
        return sorted(out) if out else None
    except Exception as e:
        print(f"[WARN] 交易日历获取失败(跳过断档校验): {e}")
        return None


def expected_last_trading_day(cal):
    """最后一个"此刻本应已经入库"的交易日 (15:05 前不算当天, 盘中/盘前不误报)。"""
    if not cal:
        return None
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    deadline = now.normalize() if now.time() > pd.Timestamp("15:05").time() \
        else now.normalize() - pd.Timedelta(days=1)
    prior = [d for d in cal if d <= deadline]
    return prior[-1] if prior else None


def missing_trading_days(dates, cal, start, end):
    if cal is None or end is None:
        return None
    have = set(pd.to_datetime(dates).dt.normalize())
    return [d for d in cal if start <= d <= end and d.normalize() not in have]


def check(results, name, ok, detail, warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append((tag, name, detail, warn))
    print(f"[{tag}] {name}: {detail}")


def main():
    repo = find_repo()
    if not repo or not os.path.isdir(repo):
        print("找不到 quant-lab 仓库, 请传入根目录参数")
        sys.exit(1)
    kdir, mdir = f"{repo}/data/kline", f"{repo}/data/meta"
    res = []
    print(f"仓库: {repo}\n")

    # ---- meta ----
    for name in META_REQUIRED:
        p = f"{mdir}/{name}.parquet"
        if os.path.exists(p):
            d = pd.read_parquet(p)
            tail = f", 最新 {pd.to_datetime(d['date']).max():%Y-%m-%d}" if "date" in d.columns else ""
            check(res, f"meta/{name}", len(d) > 0, f"{len(d)} 行{tail}")
        else:
            check(res, f"meta/{name}", False, "文件缺失")

    basic_path = f"{mdir}/stock_basic.parquet"
    if os.path.exists(basic_path):
        b = pd.read_parquet(basic_path)
        check(res, "股票池", "status" in b.columns,
              f"共 {len(b)} 只 (上市 {(b['status'] == '1').sum()}, 退市 {(b['status'] == '0').sum()})")

    idx = pd.read_parquet(f"{mdir}/index_daily.parquet")
    idx_last = pd.to_datetime(idx["date"]).max()

    # ---- kline families ----
    loaded = {}
    for fam in FAMILIES:
        df, n_shards = load_family(kdir, fam)
        loaded[fam] = df
        if df is None:
            check(res, f"kline/{fam}", False, "无任何分片")
            continue
        dup = df.duplicated(["code", "date"]).sum()
        last = df["date"].max()
        ahead = (last - idx_last).days  # 正=K线领先指数(指数缺日), 负=K线落后指数
        if ahead > 0:
            ok, detail, warn = False, (f"{n_shards} 分片 {df['code'].nunique()} 只, 最新 {last:%Y-%m-%d} "
                                       f"领先指数 {ahead} 天 -> 指数快照缺日, 需补指数"), True
        elif ahead < -4:
            ok, detail, warn = False, (f"{n_shards} 分片 {df['code'].nunique()} 只, 最新 {last:%Y-%m-%d} "
                                       f"落后指数 {-ahead} 天 -> 需补K线"), False
        else:
            ok, detail, warn = True, (f"{n_shards} 分片 {df['code'].nunique()} 只, 最新 {last:%Y-%m-%d}, "
                                      f"与指数对齐"), False
        if dup > 0:
            ok, warn = False, False
            detail += f"; (code,date) 重复 {dup} 行 -> 需清理"
        check(res, f"kline/{fam}", ok, detail, warn)

    # ---- 交易日连续性 (核心: max(date) 对得上 ≠ 中间没断档) ----
    spans = [(f, d) for f, d in loaded.items() if d is not None]
    if spans:
        cal_start = min(d["date"].min() for _, d in spans)
        cal = trading_days(cal_start, pd.Timestamp.now().normalize())
        expect_end = expected_last_trading_day(cal)
        for fam, d in spans:
            miss = missing_trading_days(d["date"], cal, cal_start, expect_end)
            if miss is None:
                continue
            head = ", ".join(f"{x:%Y-%m-%d}" for x in miss[:5])
            check(res, f"kline/{fam} 连贯性", not miss,
                  (f"{cal_start:%Y-%m-%d}~{expect_end:%Y-%m-%d} 内缺 {len(miss)} 个交易日 "
                   f"(前5: {head}) -> 跑 scripts/backfill_gap.py 回补")
                  if miss else f"{cal_start:%Y-%m-%d}~{expect_end:%Y-%m-%d} 无断档")
        miss_idx = missing_trading_days(idx["date"], cal, cal_start, expect_end)
        if miss_idx:
            check(res, "meta/index_daily 连贯性", False,
                  f"缺 {len(miss_idx)} 个交易日 (前5: "
                  + ", ".join(f"{x:%Y-%m-%d}" for x in miss_idx[:5]) + ") -> 指数缺日, K线会被误判对齐",
                  True)

    # ---- 增量/修复目录 ----
    inc_n = len(glob.glob(f"{kdir}/incremental/raw_*.parquet"))
    fix_n = len(glob.glob(f"{kdir}/fixup/*.parquet"))
    print(f"\n增量快照 {inc_n} 天, fixup 修复 {fix_n} 个文件")

    n_fail = sum(1 for t, _, _, w in res if t == "FAIL" and not w)
    n_warn = sum(1 for t, _, _, w in res if w)
    print(f"\n==== 结果: {len(res) - n_fail - n_warn} PASS / {n_fail} FAIL / {n_warn} WARN ====")
    print("FAIL=必须处理 (缺文件/重复行/数据落后>4天); WARN=指数快照缺日, 用 reference 里的补指数 procedure 修复。")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
