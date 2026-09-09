"""文档口径一致性门禁 (防止文档/代码漂移第三次复发)。"""
from pathlib import Path

import refresh_docs

ROOT = Path(__file__).resolve().parent.parent


class TestDocConsistency:
    def test_check_passes(self):
        """SCORE_MODE / MAX_HOLD 与文档声明一致 —— 不一致会 SystemExit。"""
        refresh_docs.check_doc_consistency()   # 不抛即通过

    def test_main_table_structure(self):
        t = refresh_docs.render_main_table()
        assert "开 (默认)" in t and "|" in t

    def test_handoff_auto_kpi_anchor_unique(self):
        handoff = (ROOT / "HANDOFF.md").read_text()
        assert handoff.count(refresh_docs.ANNO_START) == 1, "AUTO-KPI 起始锚点应恰好一处"
        assert handoff.count(refresh_docs.ANNO_END) == 1, "AUTO-KPI 结束锚点应恰好一处"

    def test_auto_kpi_block_has_current_numbers(self):
        """AUTO-KPI 锚点区块内必须是当前口径数字 (+114.1%), 不允许残留旧口径 (+115.3%)。
        (台账里的历史 A/B 数字合法保留, 故只检查锚点区块。)"""
        import re
        handoff = (ROOT / "HANDOFF.md").read_text()
        m = re.search(rf"{re.escape(refresh_docs.ANNO_START)}.*?{re.escape(refresh_docs.ANNO_END)}",
                      handoff, flags=re.S)
        assert m, "AUTO-KPI 区块缺失"
        block = m.group(0)
        assert "+115.3%" not in block, "锚点区块残留退市精确剔除前的旧数字 +115.3%"
        assert "夏普1.45" not in block and "1.45" not in block, "锚点区块残留旧夏普 1.45 (应为 1.44)"
        assert "+114.1%" in block, "锚点区块缺少当前口径数字"
