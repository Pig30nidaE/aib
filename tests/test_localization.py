"""Exercise language switching and the synthetic fallback with Streamlit's test runner."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from translations import translate_indicator

ROOT = Path(__file__).resolve().parents[1]


class LocalizationTests(unittest.TestCase):
    def test_case_search_and_filter_reset_across_languages(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        selected = app.session_state["selected_case"]
        app.number_input(key="queue_page_widget").set_value(2).run()
        app.text_input(key="case_search").input(selected.lower()).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["queue_page"], 1)
        self.assertEqual(
            [button.key for button in app.button if button.key and button.key.startswith("select_")],
            [f"select_{selected}"],
        )
        app.selectbox(key="language").set_value("ko").run()
        self.assertEqual(app.text_input(key="case_search").value, selected.lower())
        app.button(key="nav_catalog").click().run()
        app.button(key="nav_queue").click().run()
        self.assertEqual(app.text_input(key="case_search").value, selected.lower())
        app.text_input(key="case_search").input("no-such-case-[]").run()
        self.assertFalse(any(button.key and button.key.startswith("select_") for button in app.button))
        self.assertEqual(app.session_state["selected_case"], selected)
        app.button(key="clear_queue_filters").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="case_search").value, "")
        self.assertEqual(app.selectbox(key="category_filter_ko").value, "All categories")
        app.selectbox(key="language").set_value("en").run()
        self.assertEqual(app.selectbox(key="category_filter_en").value, "All categories")
        self.assertEqual(app.session_state["selected_case"], selected)

    def test_catalog_search_and_pagination(self):
        if not (ROOT / "fashion-dataset" / "styles.csv").exists():
            self.skipTest("Download the high-resolution fashion dataset to test catalog browsing")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.button(key="nav_catalog").click().run()
        self.assertFalse(app.exception)
        page_one = {button.key for button in app.button if button.key and button.key.startswith("product_")}
        self.assertEqual(len(page_one), 24)
        app.number_input(key="catalog_page_widget").set_value(2).run()
        page_two = {button.key for button in app.button if button.key and button.key.startswith("product_")}
        self.assertEqual(len(page_two), 24)
        self.assertFalse(page_one & page_two)
        item = next(iter(page_two))
        app.button(key=item).click().run()
        self.assertEqual(app.session_state["catalog_selected"], item.removeprefix("product_"))
        app.text_input(key="catalog_search").input("backpack").run()
        self.assertEqual(app.session_state["catalog_page"], 1)
        self.assertTrue(any("6 of 6 products displayed" in caption.value for caption in app.caption))
        selected_after_search = app.session_state["catalog_selected"]
        app.selectbox(key="language").set_value("ko").run()
        self.assertEqual(app.session_state["screen"], "catalog")
        self.assertEqual(app.session_state["catalog_search"], "backpack")
        self.assertEqual(app.session_state["catalog_selected"], selected_after_search)
        self.assertFalse(app.exception)
        app.button(key="nav_queue").click().run()
        app.button(key="nav_catalog").click().run()
        self.assertEqual(app.session_state["catalog_search"], "backpack")
        self.assertEqual(app.session_state["catalog_selected"], selected_after_search)

    def test_review_state_and_sort_survive_switches(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["language"], "en")
        self.assertTrue(any("Browseable investigation cases: 1250" in item.value for item in app.caption))
        self.assertTrue(pd.read_csv(ROOT / "archive" / "counterfeit_products.csv")["product_id"].is_unique)
        self.assertEqual(app.dataframe[1].value.to_numpy().tolist(), [[879, 4], [12, 355]])
        case_id = app.session_state["selected_case"]
        app.button(key="action_Information requested").click().run()
        app.selectbox(key="status_filter_en").set_value("Information requested").run()
        self.assertTrue(any("1 of 1 cases displayed" in item.value for item in app.caption))
        self.assertEqual(app.session_state["decisions"][case_id], "Information requested")
        app.selectbox(key="status_filter_en").set_value("All statuses").run()
        app.selectbox(key="queue_sort_en").set_value("Review status").run()
        original_log = list(app.session_state["log"])
        app.number_input(key="queue_page_widget").set_value(2).run()
        self.assertEqual(app.session_state["selected_case"], case_id)
        self.assertEqual(app.session_state["decisions"][case_id], "Information requested")
        app.selectbox(key="language").set_value("ko").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_case"], case_id)
        self.assertEqual(app.session_state["decisions"][case_id], "Information requested")
        self.assertEqual(app.session_state["sort_choice"], "Review status")
        self.assertEqual(app.selectbox(key="queue_sort_ko").value, "Review status")
        self.assertEqual(app.session_state["log"], original_log)
        self.assertTrue(any("검토 상태: 추가 정보 요청됨" in item.value for item in app.caption))
        app.button(key="nav_catalog").click().run()
        app.button(key="nav_queue").click().run()
        self.assertEqual(app.session_state["queue_page"], 2)
        app.button(key="action_Investigate").click().run()
        self.assertEqual(app.session_state["log"][-1]["human_decision"], "Investigate")
        self.assertEqual(app.session_state["log"][-1]["score_type"], "Experimental screening score")
        app.selectbox(key="language").set_value("en").run()
        self.assertEqual(app.session_state["selected_case"], case_id)
        self.assertEqual(app.session_state["decisions"][case_id], "Investigate")
        self.assertEqual(app.selectbox(key="queue_sort_en").value, "Review status")
        app.button(key="action_Dismissed").click().run()
        self.assertEqual(app.session_state["decisions"][case_id], "Dismissed")
        self.assertEqual(set(app.session_state["log"][-1]), {"timestamp_utc", "case_id", "source_product_id", "source", "score_type", "score", "suggested_next_step", "human_decision"})
        self.assertFalse(app.exception)

    def test_fallback_and_dynamic_warnings(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            app_file = Path(temp) / "app.py"
            shutil.copy(ROOT / "app.py", app_file)
            app = AppTest.from_file(str(app_file), default_timeout=30).run()
            app.selectbox(key="language").set_value("ko").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("규칙 기반" in item.value for item in app.warning))
            self.assertTrue(any("일치하는 참고 이미지 없음" in item.value for item in app.markdown))
            app.button(key="select_T9-005").click().run()
            self.assertTrue(any("가격 정보가 없습니다" in item.value for item in app.markdown))
            self.assertEqual(app.session_state["selected_case"], "T9-005")
        self.assertEqual(translate_indicator("ko", "Seller has 7 reviews."), "판매자 리뷰가 7건 있습니다.")
        self.assertEqual(translate_indicator("ko", "Return policy is marked unclear."), "반품 정책이 불명확한 것으로 표시되어 있습니다.")


if __name__ == "__main__":
    unittest.main()
