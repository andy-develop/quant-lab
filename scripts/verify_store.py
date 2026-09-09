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

FAMILIES = ["raw", "qfq", "hfq"]  # K线三种口径的分片前缀
META_REQUIRED = ["stock_basic", "index_daily", "bench_daily", "csi1000_daily"]


def find_repo():
    if len(sys.argv) > 1:
        return sys.argv[1]
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
    for fam in FAMILIES:
        df, n_shards = load_family(kdir, fam)
        if df is None:
            check(res, f"kline/{fam}", False, "无任何分片")
            continue
        dup = df.duplicated(["code", "date"]).sum()
        last = df["date"].max()
        ahead = (last - idx_last).days  # 正=K线领先指数(指数缺日), 负=K线落后指数
        if fam == "qfq":
            # qfq 已弃用: 信号链路全用 hfq, daily_update 不再给 qfq 做增量,
            # 落后属预期, 只查重复行
            ok, detail, warn = True, (f"{n_shards} 分片 {df['code'].nunique()} 只, 最新 {last:%Y-%m-%d} "
                                      f"(已弃用, 不再增量, 仅存档)"), False
        elif ahead > 0:
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
