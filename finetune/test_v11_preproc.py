import unittest
from v11_preproc import clean_text, build_hint, apply_postprocess


class TestV11Preproc(unittest.TestCase):
    def test_clean_text_drops_boilerplate(self):
        jd = "We value diversity and inclusion. Salary £70,000 - £80,000. Hybrid in London."
        cleaned = clean_text(jd)
        self.assertIn("Salary £70,000 - £80,000", cleaned)
        self.assertIn("Hybrid in London", cleaned)
        self.assertNotIn("diversity", cleaned.lower())

    def test_build_hint_basic(self):
        jd = "Remote role anywhere in UK. Salary £120k. Node.js and React required. AI/ML experience."
        hint = build_hint(jd)
        self.assertIn("loc=UK_OTHER", hint)
        self.assertIn("arr=REMOTE", hint)
        self.assertIn("comp=ABOVE_100K", hint)
        self.assertIn("AI_ML", hint)
        self.assertIn("NODE", hint)
        self.assertIn("REACT", hint)

    def test_postprocess_currency_and_tech(self):
        tokens = {"loc": "UK_OTHER", "arr": "REMOTE", "sen": "LEVEL_2", "tech": ["NODE"], "comp": "RANGE_45_54K"}
        text = "US based role paying $150k working with AI/ML"
        adjusted = apply_postprocess(tokens, text)
        self.assertEqual(adjusted["comp"], "NO_GBP")
        self.assertEqual(adjusted["loc"], "OUTSIDE_UK")
        self.assertIn("AI_ML", adjusted["tech"])

    def test_postprocess_high_salary_day_rate(self):
        tokens = {"loc": "IN_LONDON", "arr": "HYBRID", "sen": "LEVEL_3", "tech": ["JS_TS"], "comp": "RANGE_75_99K"}
        text = "£650 per day in London, TypeScript"
        adjusted = apply_postprocess(tokens, text)
        self.assertEqual(adjusted["comp"], "UP_TO_ONLY")


if __name__ == "__main__":
    unittest.main()
