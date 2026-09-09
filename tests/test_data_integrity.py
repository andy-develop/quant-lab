"""真实数据完整性 (CI checkout 自带 data/, 本地跑同样有效):
ST 逐日过滤 / 退市 out_date 精确剔除 / 状态机值域 / 净值起点归一。
"""
import pandas as pd
import pytest

from engine import BASE


@pytest.fixture(scope="module")
def sigs():
    df = pd.read_parquet(f"{BASE}/data/meta/signals.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


@pytest.fixture(scope="module")
def st_flagged():
    st = pd.read_parquet(f"{BASE}/data/meta/st_history.parquet",
                         columns=["code", "date", "is_st"])
    st = st[st["is_st"] == 1]
    st["date"] = pd.to_datetime(st["date"]).dt.normalize()
    return set(zip(st["code"], st["date"]))


class TestStFilter:
    def test_no_signal_on_st_dates(self, sigs, st_flagged):
        """信号日处于 isST==1 的 (code,date) 必须为 0 —— ST 未来函数修复的回归测试。"""
        pairs = set(zip(sigs["code"], sigs["date"]))
        leak = pairs & st_flagged
        assert not leak, f"发现 {len(leak)} 条 ST 日信号泄漏, 例: {list(leak)[:3]}"


class TestDelistFilter:
    def test_no_signal_after_out_date_window(self, sigs):
        """退市股 out_date-30 天之后不允许再有信号 (健康期保留)。"""
        basic = pd.read_parquet(f"{BASE}/data/meta/stock_basic.parquet",
                                columns=["secid", "out_date"])
        ret = basic[basic["out_date"].notna()]
        merged = sigs.merge(ret, left_on="code", right_on="secid", how="inner")
        if merged.empty:
            pytest.skip("信号中无退市股 (正常)")
        bad = merged[pd.to_datetime(merged["date"]) >=
                     pd.to_datetime(merged["out_date"]) - pd.Timedelta(days=30)]
        assert bad.empty, f"{len(bad)} 条信号落在退市窗口内, 例: {bad[['code','date']].head(3).to_dict('records')}"


class TestRegime:
    def test_target_ratio_in_range(self):
        r = pd.read_parquet(f"{BASE}/data/meta/market_regime.parquet")
        assert r["target_ratio"].dropna().between(0, 1).all()
        assert len(r) > 100


class TestEquity:
    @pytest.mark.parametrize("meta_dir,tag", [("meta", "开"), ("meta_no", "关")])
    def test_starts_at_1m(self, meta_dir, tag):
        eq = pd.read_csv(f"{BASE}/data/{meta_dir}/equity.csv")
        assert eq["equity"].iloc[0] == pytest.approx(1_000_000.0), f"{tag}口径起点未归一 100 万"
        assert len(eq) > 100

    def test_signals_sorted_by_date(self, sigs):
        """signals 按日期升序存储 (引擎按日扫描依赖)。"""
        assert sigs["date"].is_monotonic_increasing
