import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


GATE = Path(__file__).with_name("gate.py")


class TypedErasedGateTest(unittest.TestCase):
    def run_gate(self, rows, *options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("case", "variant", "threads", "median_ns", "samples"))
                writer.writerows(rows)
            return subprocess.run(
                [sys.executable, str(GATE), str(path), *options],
                text=True, capture_output=True, check=False,
            )

    def test_opt_in_and_boundaries(self):
        rows = [(case, variant, 1, ns, 11)
                for case, typed in (("below", 124), ("equal", 125), ("above", 126))
                for variant, ns in (("typed", typed), ("erased", 100))]
        default = self.run_gate(rows)
        self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
        self.assertNotIn("(b2)", default.stdout)
        enabled = self.run_gate(rows, "--typed-erased-check")
        self.assertEqual(enabled.returncode, 1, enabled.stdout + enabled.stderr)
        self.assertIn("(b2) above 1T: typed/erased = 1.260 > 1.25", enabled.stdout)
        self.assertNotIn("(b2) below", enabled.stdout)
        self.assertNotIn("(b2) equal", enabled.stdout)
        self.assertEqual(self.run_gate(rows, "--typed-erased-check", "--report-only").returncode, 0)
        self.assertEqual(self.run_gate(rows, "--typed-erased-check", "--typed-erased", "1.3").returncode, 0)
        self.assertEqual(self.run_gate(rows, "--typed-erased", "1.0").returncode, 0)

    def test_minimum_applies_to_typed_numerator(self):
        rows = [("x", "typed", 1, 126, 11), ("x", "erased", 1, 100, 11)]
        for cutoff, expected in (("110", 1), ("126", 1), ("127", 0)):
            result = self.run_gate(rows, "--typed-erased-check", "--min-ns", cutoff)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            self.assertEqual("(b2) x" in result.stdout, expected == 1)

    def test_only_exact_variant_and_thread_pairs(self):
        rows = [("x", "typed", 1, 126, 11), ("x", "erased", 4, 100, 11),
                ("y", "typed", 1, 126, 11), ("y", "erased_uninit", 1, 100, 11)]
        result = self.run_gate(rows, "--typed-erased-check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("  (b2) ", result.stdout)

    def test_zero_erased_median_is_flagged(self):
        rows = [("x", "typed", 1, 126, 11), ("x", "erased", 1, 0, 11)]
        result = self.run_gate(rows, "--typed-erased-check", "--report-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("(b2) x 1T: typed/erased = inf", result.stdout)

    def test_existing_flags_and_last_row_wins(self):
        rows = [("x", "typed", 1, 100, 11), ("x", "typed", 1, 126, 11),
                ("x", "erased", 1, 100, 11), ("x", "erased_uninit", 1, 200, 11)]
        result = self.run_gate(rows, "--typed-erased-check")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("(b) x 1T: erased_uninit/typed", result.stdout)
        self.assertEqual(result.stdout.count("(b2) x 1T:"), 1)
        self.assertIn("bb2", result.stdout)


if __name__ == "__main__":
    unittest.main()
