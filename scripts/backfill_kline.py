#!/usr/bin/env python3
"""历史K线回补: 腾讯 fqkline, 2023-09-01 起, 全A(含退市) 5552 只。
不复权(bfq) + 前复权(qfq) 双序列, 分片 parquet, 断点续拉。
用法: python backfill_kline.py [--threads 4]
"""
import datetime
import json
import os
import queue
import sys
import threading
import time

import pandas as pd
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
KDIR = f"{BASE}/data/kline"
META = f"{BASE}/data/meta"
BEG, COUNT = "2023-09-01", "800"
END = datetime.date.today().strftime("%Y-%m-%d")  # 动态截止, 避免硬编码过期后回补出静默缺口
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
      "Referer": "https://gu.qq.com/"}
COLS = ["date", "open", "close", "high", "low", "volume"]


def make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update(UA)
    return s


def fetch_kline(s: requests.Session, sym: str, fq: str) -> pd.DataFrame | None:
    """fq: '' 不复权 / 'qfq' 前复权. 返回 DataFrame 或 None."""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    for attempt in range(3):
        try:
            r = s.get(url, params={"param": f"{sym},day,{BEG},{END},{COUNT},{fq}"}, timeout=8)
            d = r.json()["data"][sym]
            key = "qfqday" if fq == "qfq" else "day"
            bars = d.get(key) or d.get("day")
            if not bars:
                return None
            recs = []
            for b in bars:
                if isinstance(b, list) and len(b) >= 6:
                    recs.append(b[:6])
            df = pd.DataFrame(recs, columns=COLS)
            df["date"] = pd.to_datetime(df["date"])
            for c in COLS[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna(subset=["close"]).reset_index(drop=True)
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1.0 + attempt * 2)
    return None


def to_tencent(secid: str) -> str:
    mkt, num = secid.split(".")
    return ("sh" if mkt == "1" else "sz") + num


def fetch_index(s: requests.Session, sym: str) -> pd.DataFrame | None:
    df = fetch_kline(s, sym, "")
    if df is not None:
        df = df.rename(columns={"volume": "volume"})
    return df


def main() -> None:
    n_threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
    basic = pd.read_parquet(f"{META}/stock_basic.parquet")
    secids = basic["secid"].tolist()
    # 断点续拉
    done: set[str] = set()
    prog_path = f"{META}/backfill_progress.json"
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            done = set(json.load(f))
    # baostock 回补已完成的也跳过 (bs_progress_*.json 里是 sh.600000 格式)
    import glob
    for pf in glob.glob(f"{META}/bs_progress_*.json"):
        with open(pf) as f:
            for c in json.load(f):
                mkt, num = c.split(".")
                done.add(("1." if mkt == "sh" else "0.") + num)
    todo = [x for x in secids if x not in done]
    print(f"总 {len(secids)}, 已完成 {len(done)}, 待拉 {len(todo)}", flush=True)

    s0 = make_session()
    # 指数: 上证指数 + 沪深300 (交易日历 + 基准)
    for sym, name in [("sh000001", "index_daily"), ("sh000300", "bench_daily")]:
        p = f"{META}/{name}.parquet"
        if not os.path.exists(p):
            df = fetch_index(s0, sym)
            if df is not None:
                df.to_parquet(p, index=False)
                print(f"{name}: {len(df)} 条 {df['date'].min().date()}~{df['date'].max().date()}", flush=True)
            else:
                print(f"{name}: 获取失败", flush=True)
        time.sleep(0.3)

    task_q: queue.Queue = queue.Queue()
    for x in todo:
        task_q.put(x)
    failed: list[str] = []
    fail_lock = threading.Lock()
    SHARD = 500
    shard_no = len([f for f in os.listdir(KDIR) if f.startswith("raw_")]) if os.path.exists(KDIR) else 0
    buf: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    buf_lock = threading.Lock()
    counter = [0]

    def flush(force: bool = False) -> None:
        nonlocal shard_no
        if not buf:
            return
        if len(buf) < SHARD and not force:
            return
        raws, qfqs = [], []
        for secid, rdf, qdf in buf:
            rdf.insert(0, "code", secid)
            qdf.insert(0, "code", secid)
            raws.append(rdf)
            qfqs.append(qdf)
        pd.concat(raws).to_parquet(f"{KDIR}/raw_{shard_no:03d}.parquet", index=False)
        pd.concat(qfqs).to_parquet(f"{KDIR}/qfq_{shard_no:03d}.parquet", index=False)
        shard_no += 1
        buf.clear()
        with open(prog_path, "w") as f:
            json.dump(sorted(done), f)
        print(f"[进度] {counter[0]}/{len(todo)} 完成, 分片 {shard_no}", flush=True)

    def worker(worker_id: int) -> None:
        s = make_session()
        consecutive_fail = 0
        while True:
            try:
                secid = task_q.get_nowait()
            except queue.Empty:
                return
            sym = to_tencent(secid)
            rdf = fetch_kline(s, sym, "")
            time.sleep(0.35)
            qdf = fetch_kline(s, sym, "qfq")
            time.sleep(0.35)
            if rdf is None or qdf is None:
                consecutive_fail += 1
                with fail_lock:
                    failed.append(secid)
                if consecutive_fail >= 10:
                    print(f"[worker-{worker_id}] 连续失败 {consecutive_fail}, 休眠 60s 降温", flush=True)
                    time.sleep(60)
                    consecutive_fail = 0
            else:
                consecutive_fail = 0
                with buf_lock:
                    done.add(secid)
                    buf.append((secid, rdf, qdf))
                    counter[0] += 1
                    flush()
            task_q.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flush(force=True)
    with open(f"{META}/backfill_failed.json", "w") as f:
        json.dump(sorted(failed), f)
    print(f"回补结束: 成功 {len(done)}, 失败 {len(failed)}, 分片 {shard_no}", flush=True)


if __name__ == "__main__":
    main()
