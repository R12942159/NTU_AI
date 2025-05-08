import sys
import os
import re
from typing import Tuple
from main import main


class TerminalColors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'


def get_pyautogen_version() -> Tuple[bool, str]:
    """Return (is_installed, version)."""
    try:
        try:
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:  # Python < 3.8
            from importlib_metadata import version, PackageNotFoundError
        return True, version("pyautogen")
    except PackageNotFoundError:
        return False, ""


def check_pyautogen_version(expected: str = "0.9.0") -> bool:
    """Check pyautogen version and print result."""
    present, ver = get_pyautogen_version()
    if not present:
        print(f"{TerminalColors.RED}[X] Test 0 Failed. Pyautogen missing.{TerminalColors.RESET}")
        return False
    if ver == expected:
        print(f"{TerminalColors.GREEN}[V] Test 0 Passed. Pyautogen {ver}{TerminalColors.RESET}")
        return True
    print(
        f"{TerminalColors.RED}pyautogen version mismatch — installed: {ver} expected: {expected}{TerminalColors.RESET}"
    )
    return False


# ─────────────────────────────────────────────
# Configuration: data file path
# ─────────────────────────────────────────────
# Allow the tester to specify a custom restaurant‑review data file.
# Usage: python test.py path/to/data.txt
DATA_FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else "restaurant-data.txt"


def contains_num_with_tolerance(text: str, target: float, tol: float) -> Tuple[bool, float]:
    """
    Return (passed, best_match).
    Accept any int/float; pattern widened to \d+(\.\d+)?.
    """
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return False, None
    pred = min(nums, key=lambda x: abs(x - target))
    return abs(pred - target) <= tol, pred


def public_tests() -> None:
    queries = [
        "How good is the restaurant taco bell overall?",
        "How good is the restaurant Chick-fil-A overall?",
        "What is the overall score for Starbucks?",
        "What is the overall score for In-n-Out",
        "What is the overall score for McDonald's?",
    ]
    expected = [3.00, 9.35, 8.06, 9.54, 3.65]
    tolerances = [0.20, 0.20, 0.15, 0.15, 0.15]

    logs, abs_errors, passed = [], [], 0

    for q in queries:
        # capture stdout
        with open("runtime-log.txt", "w") as f:
            sys.stdout = f
            main(q, DATA_FILE_PATH)  # pass data file path explicitly
        sys.stdout = sys.__stdout__
        with open("runtime-log.txt") as f:
            logs.append(f.read())

    for i, log in enumerate(logs):
        ok, pred = contains_num_with_tolerance(log, expected[i], tolerances[i])

        # compute error with penalty rule
        error_cap = 10.0  # maximum penalty for extremely wrong answers

        if pred is None:
            # no answer → full penalty
            abs_errors.append(error_cap)
        elif ok:  # within tolerance
            abs_errors.append(abs(pred - expected[i]))
        else:
            # answered but wrong → real error, capped
            abs_errors.append(min(abs(pred - expected[i]), error_cap))

        if ok:
            print(
                f"{TerminalColors.GREEN}[V] Test {i+1} Passed.{TerminalColors.RESET} "
                f"Expected: {expected[i]:.3f}  Predicted: {pred:.3f}  "
                f"Query: {queries[i]}"
            )
            passed += 1
        else:
            print(
                f"{TerminalColors.RED}[X] Test {i+1} Failed.{TerminalColors.RESET} "
                f"Expected: {expected[i]:.3f}  Predicted: {pred}  "
                f"Query: {queries[i]}"
            )

    mae = sum(abs_errors) / len(abs_errors)
    print(f"---------- {passed}/{len(queries)} Tests Passed. MAE: {mae:.4f} ----------")


if __name__ == "__main__":
    check_pyautogen_version("0.9.0")
    public_tests()
