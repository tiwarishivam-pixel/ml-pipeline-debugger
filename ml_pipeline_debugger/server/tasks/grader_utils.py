"""
Shared grader utilities.

All graders use the same pattern:
  1. Run fixed_code in a subprocess with a timeout
  2. Parse 'loss:X.XXXXXX' lines from stdout
  3. Score mathematically — no LLM, no fuzzy matching

This guarantees deterministic, reproducible scores every time.
"""

import math
import re
import subprocess
import sys
from typing import List, Optional, Tuple


# ── subprocess runner ──────────────────────────────────────────────────────────

def run_script(code: str, timeout: int = 45) -> Tuple[bool, str, str]:
    """
    Execute a Python script in an isolated subprocess.

    Returns:
        (success, stdout, stderr)
        success=False if the process crashed or timed out.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired: script exceeded time limit"
    except Exception as exc:
        return False, "", f"SubprocessError: {exc}"


# ── loss parser ────────────────────────────────────────────────────────────────

def parse_losses(stdout: str) -> List[float]:
    """
    Extract loss values from stdout.

    Training scripts must print: loss:X.XXXXXX
    Example line:  loss:0.234561

    Returns a list of floats. NaN/Inf are preserved as float('nan') / float('inf').
    """
    losses = []
    for match in re.finditer(r"loss:([\d.naninf+-]+)", stdout, re.IGNORECASE):
        try:
            val = float(match.group(1))
        except ValueError:
            val = float("nan")
        losses.append(val)
    return losses


# ── loss analysis helpers ──────────────────────────────────────────────────────

def has_nan(losses: List[float]) -> bool:
    return any(math.isnan(v) or math.isinf(v) for v in losses)


def is_decreasing(losses: List[float], window: int = 5) -> bool:
    """
    Check if the loss trend is downward over the last `window` steps.
    Uses a simple linear regression slope on the last window values.
    """
    clean = [v for v in losses if not math.isnan(v) and not math.isinf(v)]
    if len(clean) < window:
        return False
    tail = clean[-window:]
    n = len(tail)
    mean_x = (n - 1) / 2
    mean_y = sum(tail) / n
    numerator = sum((i - mean_x) * (tail[i] - mean_y) for i in range(n))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return False
    slope = numerator / denominator
    return slope < 0


def pct_reduction(losses: List[float]) -> Optional[float]:
    """
    Return percentage reduction from first to last valid loss value.
    Returns None if not enough valid values.
    """
    clean = [v for v in losses if not math.isnan(v) and not math.isinf(v)]
    if len(clean) < 2 or clean[0] == 0:
        return None
    return (clean[0] - clean[-1]) / clean[0]
