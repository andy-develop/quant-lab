"""大盘仓位状态机 market_regime 的合成序列单测。

Z0/Z1/Z2/Z3 转移是纯确定性逻辑, 此前零测试 —— 改一行就可能让全部 A/B 结论失效。
用可手算的合成指数序列锁定核心语义:
  - 均线未成形 (NaN 预热) -> Z1
  - 横盘 -> 维持 Z1
  - 均线多头排列 (c>MA5>MA10>MA20) -> Z3
  - 多头中单日回踩 (c 仍 >MA20 但失守 MA5) -> Z2
  - 持续急跌 (MA5<MA10<MA20) -> Z0 逃顶
  - target_ratio 映射唯一且在 {0, 0.3, 0.5, 1.0}
"""
import pandas as pd
import pytest

import signals

Z_RATIO = {"Z0": 0.0, "Z1": 0.3, "Z2": 0.5, "Z3": 1.0}


def _run_regime(closes, tmp_path):
    idx = pd.DataFrame({
        "date": pd.bdate_range("2026-01-05", periods=len(closes)),
        "close": closes,
    })
    f_in = tmp_path / "idx.parquet"
    f_out = tmp_path / "regime.parquet"
    idx.to_parquet(f_in, index=False)
    return signals.market_regime(index_file=str(f_in), out_file=str(f_out))


class TestWarmupAndFlat:
    def test_nan_warmup_is_z1(self, tmp_path):
        """MA20 未成形的预热期 (前19天) 必须是 Z1, 不得出现 NaN 状态。"""
        r = _run_regime([100.0] * 25, tmp_path)
        assert (r["state"].iloc[:19] == "Z1").all()
        assert r["state"].notna().all()

    def test_flat_market_stays_z1(self, tmp_path):
        """横盘: 所有条件都不触发, 维持初始 Z1。"""
        r = _run_regime([100.0] * 40, tmp_path)
        assert (r["state"] == "Z1").all()
        assert (r["target_ratio"] == 0.3).all()


class TestBullAndPullback:
    def test_uptrend_reaches_z3(self, tmp_path):
        """25 天横盘 + 10 天每日 +1% -> 尾段均线多头排列 -> Z3。"""
        closes = [100.0] * 25
        p = 100.0
        for _ in range(10):
            p *= 1.01
            closes.append(p)
        r = _run_regime(closes, tmp_path)
        assert r["state"].iloc[-1] == "Z3"
        assert r["target_ratio"].iloc[-1] == 1.0

    def test_single_pullback_downgrades_to_z2(self, tmp_path):
        """多头后单日 -2.5%: c 失守 MA5 但仍 >MA20 -> Z2 (有效站上20日线分支)。"""
        closes = [100.0] * 25
        p = 100.0
        for _ in range(10):
            p *= 1.01
            closes.append(p)
        closes.append(p * 0.975)          # index 35: 单日回踩
        r = _run_regime(closes, tmp_path)
        assert r["state"].iloc[35] == "Z2"
        assert r["state"].iloc[34] == "Z3"


class TestCrash:
    def test_persistent_decline_triggers_z0(self, tmp_path):
        """多头后连续 8 天 -3% -> MA5<MA10<MA20 -> Z0 逃顶, 目标仓位 0。"""
        closes = [100.0] * 25
        p = 100.0
        for _ in range(10):
            p *= 1.01
            closes.append(p)
        for _ in range(8):
            p *= 0.97
            closes.append(p)
        r = _run_regime(closes, tmp_path)
        assert r["state"].iloc[-1] == "Z0"
        assert r["target_ratio"].iloc[-1] == 0.0
        # 逃顶必须真实出现 (而非只在最后一天)
        assert "Z0" in set(r["state"].iloc[35:])


class TestRatioMapping:
    def test_ratios_exactly_four_levels(self, tmp_path):
        """target_ratio 只能取 {0, 0.3, 0.5, 1.0} 且与 state 映射一致。"""
        closes = [100.0] * 25
        p = 100.0
        for _ in range(10):
            p *= 1.01
            closes.append(p)
        for _ in range(8):
            p *= 0.97
            closes.append(p)
        r = _run_regime(closes, tmp_path)
        assert set(r["target_ratio"].unique()) <= {0.0, 0.3, 0.5, 1.0}
        assert (r["state"].map(Z_RATIO) == r["target_ratio"]).all()
