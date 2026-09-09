"""评分合成核心: KDJ_W 权重扫描开关 / pct-rank 方向与值域。

toy 数据直接喂 signal_scores (只依赖因子列), 不触发全库重算。
"""
import numpy as np
import pandas as pd
import pytest

import signals


def toy_df(n_days: int = 3, n_codes: int = 6) -> pd.DataFrame:
    codes = [f"1.60{i:04d}" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([codes, pd.date_range("2026-01-05", periods=n_days)],
                                     names=["code", "date"])
    n = len(idx)
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "ret20": rng.normal(0.02, 0.05, n),
        "sharpe20": rng.normal(0.5, 1.0, n),
        "vol20": rng.uniform(0.01, 0.05, n),
        "win60": rng.uniform(0.2, 0.9, n),
        "trend_streak": rng.integers(0, 30, n).astype(float),
        "j_dev": rng.normal(0, 10, n),
        "illiq20": rng.uniform(1, 100, n),
    }, index=idx)


@pytest.fixture
def restore_kdj_w():
    """KDJ_W 是模块级全局, 测试后必须恢复生产口径 None。"""
    yield
    signals.KDJ_W = None


class TestKdjWScan:
    def test_w0_equals_quality(self, restore_kdj_w):
        """kdj 权重=0 (从低波动匀出家族) 时 kdj5 必须严格退化为 quality 基线。"""
        signals.KDJ_W = 0.0
        s = signals.signal_scores(toy_df())
        pd.testing.assert_series_equal(s["quality_kdj5"], s["quality"], check_names=False)

    def test_w005_equals_production_default(self, restore_kdj_w):
        """KDJ_W=0.05 与生产默认 (None) 产物一致 —— 扫描家族包含生产点。"""
        default = signals.signal_scores(toy_df())["quality_kdj5"]
        signals.KDJ_W = 0.05
        forced = signals.signal_scores(toy_df())["quality_kdj5"]
        pd.testing.assert_series_equal(default, forced, check_names=False)

    def test_none_restores_production(self, restore_kdj_w):
        """KDJ_W=None (生产默认) 时 kdj5 与 quality 不同 —— kdj 仍持有 0.05 权重。"""
        signals.KDJ_W = 0.10   # 先污染
        signals.KDJ_W = None   # 再恢复
        df = toy_df()
        s = signals.signal_scores(df)
        assert not np.allclose(s["quality_kdj5"].values, s["quality"].values)


class TestScoreShape:
    def test_pct_rank_in_unit_interval(self):
        s = signals.signal_scores(toy_df())
        for name in ("quality", "quality_kdj5", "momentum", "kdj"):
            assert s[name].between(0, 1.0 + 1e-9).all(), f"{name} 超出 [0,1]"

    def test_momentum_direction(self):
        """ret20 越大 mom 越高 (同日横截面)。"""
        df = toy_df()
        mom = signals.signal_scores(df)["momentum"]
        d = df.index.get_level_values(1)[0]
        day = mom[mom.index.get_level_values(1) == d]
        ret = df["ret20"][df.index.get_level_values(1) == d]
        assert day.idxmax()[0] == ret.idxmax()[0]

    def test_kdj_direction(self):
        """j_dev 越大 (自高位回落越多) kdj 得分越高 —— 低吸方向。"""
        df = toy_df()
        kdj = signals.signal_scores(df)["kdj"]
        d = df.index.get_level_values(1)[0]
        day = kdj[kdj.index.get_level_values(1) == d]
        jd = df["j_dev"][df.index.get_level_values(1) == d]
        assert day.idxmax()[0] == jd.idxmax()[0]


class TestExitFlags:
    def test_shrink_condition(self):
        from signals import exit_flags
        df = pd.DataFrame({
            "volume": [300.0, 100.0, 300.0, 300.0],
            "prev_vol_ma5": [100.0, 100.0, 100.0, 100.0],
            "ret": [-0.01, -0.01, 0.01, -0.01],
            "close": [9.0, 9.0, 9.0, 11.0],
            "open": [10.0, 10.0, 10.0, 10.0],
        })
        flags = exit_flags(df)
        # 仅首行: 放量 + 阴线 + 收跌 同时满足
        assert list(flags.astype(bool)) == [True, False, False, False]
