import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upgraded_implementation" / "src"))

from multimodal_chart_gspo.answer_parser import extract_final_answer


class AnswerParserTests(unittest.TestCase):
    def test_tagged_and_labelled_answers(self):
        self.assertEqual("42", extract_final_answer("reasoning <answer> 42 </answer>"))
        self.assertEqual("blue", extract_final_answer("Final Answer: blue"))
        self.assertIsNone(extract_final_answer("reasoning only"))


if __name__ == "__main__":
    unittest.main()
