"""覆盖率门禁「两侧代码格式」+ 成交量单位 —— 2026-09-18 数据全丢的回归测试。

## 病根
快照行 code 是**腾讯 secid** 式 (`1.600000`), universe 是 **baostock** 式 (`sh.600000`)。
门禁原实现直接对两个字符串集合求交:

    got = df["code"].tolist()                    # 1.600000 ...
    exp = basic[basic.status == "1"]["code"]     # sh.600000 ...
    got & exp -> 恒为空

于是「快照其实拿全了 5203 只」被判成 0%, 触发红灯**拒绝写盘** —— 当天增量数据全丢。
CI 实测原文 (2026-09-18 21:01):
    [coverage/stock] RED 0/5215 = 0.0% (other=0.0%)
`other=0.0%` 是第二个症状: vendor._market_of 只认 6 位纯数字, 带前缀的代码全落进
other 桶 -> 分市场下限(§0.4 的核心防线)静默失效。

## 修法
两侧都过 `daily_update._code6()` 归一成 6 位数字码再比。
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

# daily_update 模块级 import requests; 缺 requests 时干净 skip 而不是中断整个收集
daily_update = pytest.importorskip("daily_update", reason="daily_update 依赖 requests 未安装")

DAY = pd.Timestamp("2026-09-18")


def _to_secid(code: str) -> str:
    """baostock `sh.600000` -> 腾讯 secid `1.600000` (快照的真实 code 形态)。"""
    mkt, num = code.split(".")
    return ("1." if mkt == "sh" else "0.") + num


def _universe(n_sh: int = 10, n_sz: int = 2) -> pd.DataFrame:
    """上市股票池: baostock 式 code + status=1 (与 data/meta/stock_basic.parquet 同构)。"""
    codes = [f"sh.{600000 + i}" for i in range(n_sh)] + [f"sz.{1 + i:06d}" for i in range(n_sz)]
    return pd.DataFrame({"code": codes, "status": ["1"] * len(codes)})


def _snapshot(codes) -> pd.DataFrame:
    """快照: 腾讯 secid 式 code (与 daily_update.snapshot_day 产出同构)。"""
    return pd.DataFrame({"code": list(codes)})


# ---------------------------------------------------------------- 归一函数本身
def test_code6_normalizes_every_known_form():
    assert daily_update._code6("1.600000") == "600000"      # 腾讯 secid
    assert daily_update._code6("sh.600000") == "600000"     # baostock
    assert daily_update._code6("sz.000001") == "000001"     # ★ 前导零不能丢
    assert daily_update._code6("600000") == "600000"        # 裸 6 位
    assert daily_update._code6(" 0.300750 ") == "300750"    # 带空白


def test_raw_code_formats_are_disjoint():
    """病根写真: 不归一的话两侧集合**永不相交** —— 这就是 RED 0/5215 的成因。"""
    basic = _universe()
    snap = _snapshot(_to_secid(c) for c in basic["code"])
    assert not (set(snap["code"]) & set(basic["code"]))
    # 归一后才相交, 且一只不少
    assert {daily_update._code6(c) for c in snap["code"]} == \
           {daily_update._code6(c) for c in basic["code"]}


# ---------------------------------------------------------------- 门禁行为
def test_gate_passes_when_snapshot_is_complete_despite_code_format():
    """★ 核心回归: 快照拿全了就不许判红, 哪怕两侧 code 写法不同 (修前正是这里判红)。"""
    basic = _universe()
    snap = _snapshot(_to_secid(c) for c in basic["code"])
    daily_update._gate_snapshot(snap, basic, DAY)           # 不抛异常即通过


def test_gate_still_red_when_market_really_missing():
    """反向守护: 归一化不能把门禁改成永远绿灯 —— 真的整市场缺失必须拦下。"""
    basic = _universe(n_sh=10, n_sz=2)
    snap = _snapshot(_to_secid(c) for c in basic["code"] if c.startswith("sh."))
    with pytest.raises(RuntimeError, match="门禁"):
        daily_update._gate_snapshot(snap, basic, DAY)


def test_gate_per_market_floor_alive_after_normalization():
    """整体覆盖率达标但深市为 0 -> 仍须判红 (分市场下限是 §0.4 的核心防线)。

    修前所有带前缀代码落入 other 桶, 该防线静默失效。
    """
    basic = _universe(n_sh=100, n_sz=10)                    # 整体 100/110 = 90.9% >= 80%
    snap = _snapshot(_to_secid(c) for c in basic["code"] if c.startswith("sh."))
    with pytest.raises(RuntimeError):
        daily_update._gate_snapshot(snap, basic, DAY)


# ---------------------------------------------------------------- 成交量单位
def test_to_shares_converts_hands_to_shares():
    """腾讯系 volume 单位是**手**, 库内口径是**股** -> ×100 (HANDOFF 陷阱 #4)。"""
    df = pd.DataFrame({"volume": [1234, 5678]})
    out = daily_update._to_shares(df)
    assert out["volume"].tolist() == [123400, 567800]
    assert out["volume"].dtype == "int64"


def test_to_shares_tolerates_empty_and_missing_column():
    """空表 / 无 volume 列 (如 hfq 家族) 不能炸。"""
    assert daily_update._to_shares(pd.DataFrame({"volume": []})).empty
    df = pd.DataFrame({"close": [1.0]})
    assert list(daily_update._to_shares(df).columns) == ["close"]