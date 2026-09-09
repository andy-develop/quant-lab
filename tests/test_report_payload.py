"""报告 payload (单一全期口径) 与阈值常量的回归测试。

覆盖: 区间切换归一语义唯一 (#16) / MAX_HOLD 参数化 (#11 系列) / ST 横幅 (#11)。
"""
import pandas as pd
import pytest

import build_report
import engine
import refresh_docs
from build_report import _shared, build_payload


@pytest.fixture(scope="module")
def modes():
    bench, bench1000, sigs = _shared()
    off_dir = f"{engine.BASE}/data/meta_no"
    p_on = build_payload(f"{engine.BASE}/data/meta", "on", bench, bench1000, sigs,
                         window_years=None)
    p_off = build_payload(off_dir, "off", bench, bench1000, sigs, window_years=None)
    return {"on": p_on, "off": p_off}


class TestPayload:
    def test_single_full_range_payload(self, modes):
        """全期 payload: 净值起点 100 万, 覆盖 equity.csv 全部日期 —— 归一语义唯一。"""
        p = modes["on"]
        eq = pd.read_csv(f"{engine.BASE}/data/meta/equity.csv")
        assert p["equity"][0] == pytest.approx(1_000_000.0)
        assert len(p["dates"]) == len(eq), "payload 与 equity.csv 日期数不一致"

    def test_on_off_same_dates(self, modes):
        assert modes["on"]["dates"] == modes["off"]["dates"]

    def test_st_banner_type(self):
        """ST 数据新鲜度横幅: 当前滞后小应返回空串或含'截至'提示, 都必须是 str。"""
        banner = build_report._st_banner_html()
        assert isinstance(banner, str)
        if banner:
            assert "ST 数据截至" in banner or "落后" in banner


class TestConstants:
    def test_doc_consistency_real(self):
        refresh_docs.check_doc_consistency()

    def test_max_hold_documented(self):
        assert engine.MAX_HOLD == refresh_docs.DOC_MAX_HOLD == 10

    def test_score_mode_is_kdj5(self):
        import signals
        assert signals.SCORE_MODE == "quality_kdj5"
        assert signals.KDJ_W is None, "KDJ_W 扫描开关未恢复生产口径 (必须为 None)"
