"""load_store 代码格式统一 + fixup 覆盖 (2026-09-08 冷启动事故的回归测试)。

历史分片是 baostock 式 (sh.600000), fixup 是腾讯式 (1.600000)。
顺序颠倒 / 不统一 -> fixup isin 全 miss -> 整段重复行。
"""
import pandas as pd

import daily_update


def _mk(tmp_path, sub, rows):
    import os
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "raw_x.parquet", index=False)


def _patch_kdir(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_update, "KDIR", str(tmp_path))


def test_store_code_unified_and_fixup_applied(tmp_path, monkeypatch):
    # 分片直接在 KDIR 根目录; fixup 在 KDIR/fixup 子目录 (真实布局)
    _mk(tmp_path, ".", [
        {"code": "sh.600000", "date": "2026-01-05", "close": 10.0},
        {"code": "sz.000001", "date": "2026-01-05", "close": 12.0},
    ])
    _mk(tmp_path, "fixup", [
        {"code": "1.600000", "date": "2026-01-05", "close": 10.5},   # 修复后的腾讯式
    ])
    _patch_kdir(monkeypatch, tmp_path)

    df = daily_update.load_store()

    # 无重复行 (fixup 覆盖成功, 不是追加)
    assert not df.duplicated(["code", "date"]).any()
    # 代码已统一为腾讯式
    assert set(df["code"]) == {"1.600000", "0.000001"}
    # fixup 的新值生效
    assert df.loc[df["code"] == "1.600000", "close"].iloc[0] == 10.5


def test_store_no_fixup_dir(tmp_path, monkeypatch):
    """fixup 目录不存在时不报错, 原样返回分片。"""
    _mk(tmp_path, ".", [{"code": "sh.600000", "date": "2026-01-05", "close": 10.0}])
    _patch_kdir(monkeypatch, tmp_path)
    df = daily_update.load_store()
    assert len(df) == 1 and df["code"].iloc[0] == "1.600000"
