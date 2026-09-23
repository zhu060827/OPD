import unittest

from teacher_training.test_execution import run_registered_tests


class TestExecutionTests(unittest.TestCase):
    def test_python_assert(self):
        passed, _ = run_registered_tests("def f(x):\n    return x + 1", ["assert f(1) == 2"])
        self.assertTrue(passed)

    def test_stdin_stdout(self):
        passed, _ = run_registered_tests(
            "value = int(input())\nprint(value + 1)",
            [{"input": "2\n", "output": "3\n"}], "stdin_stdout",
        )
        self.assertTrue(passed)

    def test_stdin_stdout_rejects_wrong_output(self):
        passed, _ = run_registered_tests(
            "print(input())", [{"input": "a\n", "output": "b\n"}], "stdin_stdout",
        )
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
