import unittest

from teacher_training.scripts.adapt_source_data import adapt_rows


class AdaptSourceDataTests(unittest.TestCase):
    def test_commitpackft_fields(self):
        rows, manifest = adapt_rows([{
            "commit": "abc", "file_path": "a.py", "lang": "Python",
            "new_contents": "def answer(value):\n    return value + 1\n",
        }], "commitpackft")
        self.assertEqual(1, manifest["accepted"])
        self.assertEqual("abc:a.py", rows[0]["problem_id"])

    def test_codesearchnet_fields(self):
        rows, _ = adapt_rows([{
            "repo": "r", "path": "p.py", "func_name": "f",
            "code": "def f(value):\n    result = value + 1\n    return result\n",
        }], "codesearchnet")
        self.assertEqual("code-search-net/code_search_net", rows[0]["source_dataset"])

    def test_code_contests_python_and_tests(self):
        rows, _ = adapt_rows([{
            "name": "plus-one", "description": "add one",
            "solutions": {"language": [3, 1], "solution": [
                "value = int(input().strip())\nanswer = value + 1\nprint(answer)\n", "not python"
            ]},
            "public_tests": {"input": ["2\n"], "output": ["3\n"]},
        }], "code_contests", verify_tests=True)
        self.assertEqual(1, len(rows))
        self.assertEqual("stdin_stdout", rows[0]["test_format"])


if __name__ == "__main__":
    unittest.main()
