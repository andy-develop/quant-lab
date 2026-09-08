#!/usr/bin/env python3
"""拉取全A股票池（含退市股，防幸存者偏差）+ 指数日K（交易日历 + 基准）。
输出:
  data/meta/stock_basic.parquet  -- code_bs, secid, name, ipo_date, out_date, status
  data/meta/index_daily.parquet  -- 上证指数日线（交易日历来源）
  data/meta/bench_daily.parquet  -- 沪深300日线（业绩基准）
"""
import os
import sys
import time

import pandas as pd
import requests
import baostock as bs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
BEG = "20230901"  # 回补起点: 2023-09-01, 约3年


def fetch_universe() -> pd.DataFrame:
    lg = bs.login()
    assert lg.error_code == "0", f"baostock login failed: {lg.error_msg}"
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    fields = rs.fields
    bs.logout()
    df = pd.DataFrame(rows, columns=fields)
    df = df[df["type"] == "1"].copy()  # 只要股票, 排除指数
    # 映射东财 secid: sh.600000 -> 1.600000, sz.000001 -> 0.000001
    def to_secid(code: str) -> str:
        mkt, num = code.split(".")
        return ("1." if mkt == "sh" else "0.") + num
    df["secid"] = df["code"].map(to_secid)
    df = df.rename(columns={"code_name": "name", "ipoDate": "ipo_date", "outDate": "out_date"})
    df["ipo_date"] = pd.to_datetime(df["ipo_date"])
    df["out_date"] = pd.to_datetime(df["out_date"].replace("", None))
    df = df[["code", "secid", "name", "ipo_date", "out_date", "status"]]
    df.to_parquet(f"{BASE}/data/meta/stock_basic.parquet", index=False)
    print(f"股票池: {len(df)} 只 (上市中 {(df['status']=='1').sum()}, 退市 {(df['status']=='0').sum()})")
    return df


def fetch_em_kline(secid: str, fqt: str = "0", retries: int = 3) -> pd.DataFrame | None:
    s = requests.Session()
    s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    for i in range(retries):
        try:
            r = s.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
                        "fields2": "f51,f52,f53,f54,f55,f56,f57",
                        "klt": "101", "fqt": fqt, "beg": BEG, "end": "20500101"},
                timeout=20,
            )
            d = r.json().get("data")
            if not d or not d.get("klines"):
                return None
            recs = [k.split(",") for k in d["klines"]]
            df = pd.DataFrame(recs, columns=["date", "open", "close", "high", "low", "volume", "amount"])
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "close", "high", "low", "volume", "amount"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(1.5 * (i + 1))
    return None


def main() -> None:
    df = fetch_universe()
    # 指数 + 基准 (csi1000_daily=中证1000: 大盘状态机/净值图基准)
    for secid, name in [("1.000001", "index_daily"), ("1.000300", "bench_daily"),
                        ("1.000852", "csi1000_daily")]:
        k = fetch_em_kline(secid)
        if k is not None:
            k.to_parquet(f"{BASE}/data/meta/{name}.parquet", index=False)
            print(f"{name}: {len(k)} 条, {k['date'].min().date()} ~ {k['date'].max().date()}")
        else:
            print(f"{name}: 获取失败!", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
