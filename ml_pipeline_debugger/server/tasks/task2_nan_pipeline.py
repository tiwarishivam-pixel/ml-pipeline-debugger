"""
Task 2 — Medium: Data Pipeline NaN Bomb.

The training loop and model are both fine. The bug is hidden inside the
data preprocessing function. A subtle mistake causes NaN or Inf values
to enter the model and propagate into the loss immediately.

The agent must trace the NaN backwards through the preprocessing code,
identify the exact line, and fix it. No error message — just NaN loss.

Bug variants:
  - div_by_zero_std:   normalise without checking std == 0
  - log_of_zero:       log transform without clipping values > 0
  - sqrt_negative:     sqrt without abs()
  - inf_from_scale:    feature scaled by 1e40 → overflow to Inf in float32
  - missing_fillna:    sparse feature with NaN not filled before tensor cast

Grader scoring (0.0 – 1.0, partial credit):
  +0.3  script runs without crashing
  +0.4  zero NaN values in the entire loss curve
  +0.3  loss reduces by ≥ 50% over 30 steps
"""

import random
from typing import List, Tuple

from .grader_utils import has_nan, is_decreasing, parse_losses, pct_reduction, run_script

# ── script templates ───────────────────────────────────────────────────────────

_BROKEN_SCRIPTS = {

    "div_by_zero_std": """
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def preprocess(X_np):
    # BUG: no guard against std == 0 for constant columns
    X_norm = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)
    return torch.tensor(X_norm, dtype=torch.float32)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

raw_X = np.random.randn(300, 10)
raw_X[:, 4] = 7.0          # column 4 is constant — std = 0 → division by zero
y = torch.randn(300, 1)

for epoch in range(30):
    X = preprocess(raw_X)
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",

    "log_of_zero": """
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def preprocess(X_np):
    # BUG: log of zero or negative values → -inf / NaN
    X_log = np.log(X_np)
    return torch.tensor(X_log, dtype=torch.float32)

model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Data has zeros and negatives — log will produce -inf and NaN
raw_X = np.random.randn(300, 8)   # can be negative!
y = torch.randn(300, 1)

for epoch in range(30):
    X = preprocess(raw_X)
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",

    "sqrt_negative": """
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def preprocess(X_np):
    # BUG: sqrt of negative numbers → NaN
    X_sqrt = np.sqrt(X_np)
    return torch.tensor(X_sqrt, dtype=torch.float32)

model = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

raw_X = np.random.randn(300, 6)    # has negative values → sqrt → NaN
y = torch.randn(300, 1)

for epoch in range(30):
    X = preprocess(raw_X)
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",

    "inf_from_scale": """
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def preprocess(X_np):
    # BUG: feature 2 multiplied by 1e40 → overflows float32 → Inf
    X_np = X_np.copy()
    X_np[:, 2] = X_np[:, 2] * 1e40  # overflows float32 → Inf
    return torch.tensor(X_np, dtype=torch.float32)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

raw_X = np.random.randn(300, 10)
y = torch.randn(300, 1)

for epoch in range(30):
    X = preprocess(raw_X)
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",

    "missing_fillna": """
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def preprocess(X_np):
    # BUG: sparse feature has NaN values, not filled before tensor cast
    return torch.tensor(X_np, dtype=torch.float32)

model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

raw_X = np.random.randn(300, 8)
# Simulate a sparse feature — 40% of column 1 is missing (NaN)
mask = np.random.rand(300) < 0.4
raw_X[mask, 1] = np.nan
y = torch.randn(300, 1)

for epoch in range(30):
    X = preprocess(raw_X)
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",
}

# ── descriptions ───────────────────────────────────────────────────────────────

_DESCRIPTIONS = {
    "div_by_zero_std": (
        "SYMPTOM: Loss is NaN from step 0.\n"
        "The training loop and model architecture look completely fine.\n"
        "The data has a constant-valued column. Inspect the preprocess() function carefully."
    ),
    "log_of_zero": (
        "SYMPTOM: Loss is NaN from step 0.\n"
        "The model and optimizer are correct.\n"
        "The raw data can contain zeros and negative values. "
        "Inspect the preprocess() function — specifically any mathematical transforms."
    ),
    "sqrt_negative": (
        "SYMPTOM: Loss is NaN from step 0.\n"
        "The model is fine. The raw input data contains negative values.\n"
        "Inspect preprocess() for any operations that require non-negative inputs."
    ),
    "inf_from_scale": (
        "SYMPTOM: Loss is Inf from step 0.\n"
        "The model and loop are correct.\n"
        "One feature column has an abnormally large scale. "
        "Inspect the preprocess() function for any scaling operations."
    ),
    "missing_fillna": (
        "SYMPTOM: Loss is NaN from step 0.\n"
        "The model and loop are correct.\n"
        "The dataset is sparse — some feature values are missing. "
        "Inspect preprocess() for handling of missing values before tensor conversion."
    ),
}


# ── public API ─────────────────────────────────────────────────────────────────

def sample_task2() -> Tuple[str, str, List[float], str]:
    """Randomly select a bug variant and return a task2 episode."""
    bug_key = random.choice(list(_BROKEN_SCRIPTS.keys()))
    broken_script = _BROKEN_SCRIPTS[bug_key].strip()

    _, stdout, _ = run_script(broken_script, timeout=30)
    losses = parse_losses(stdout)
    curve_preview = losses[:10]

    description = (
        f"{_DESCRIPTIONS[bug_key]}\n"
        f"Loss curve (first {len(curve_preview)} steps): {[round(l, 4) if not __import__('math').isnan(l) else 'NaN' for l in curve_preview]}"
    )
    return bug_key, broken_script, curve_preview, description


def grade_task2(fixed_code: str) -> Tuple[float, str]:
    """Grade the agent's fixed script. Returns (score 0.0-1.0, feedback)."""
    score = 0.0
    feedback_parts = []

    success, stdout, stderr = run_script(fixed_code, timeout=45)
    if not success:
        return 0.0, f"Script crashed or timed out.\nStderr: {stderr[:300]}"
    score += 0.3
    feedback_parts.append("✓ Script ran cleanly (+0.3)")

    losses = parse_losses(stdout)
    if not losses:
        return score, "\n".join(feedback_parts) + "\n✗ No loss values parsed. Print 'loss:X.XXXXXX' each epoch."

    # Core check — eliminate ALL NaN/Inf (0.4)
    if not has_nan(losses):
        score += 0.4
        feedback_parts.append("✓ No NaN or Inf in loss curve (+0.4)")
    else:
        nan_count = sum(1 for l in losses if __import__('math').isnan(l) or __import__('math').isinf(l))
        feedback_parts.append(f"✗ Loss curve still contains {nan_count} NaN/Inf values")

    # Bonus — loss actually improving (0.3)
    reduction = pct_reduction(losses)
    if reduction is not None and reduction >= 0.50:
        score += 0.3
        feedback_parts.append(f"✓ Loss reduced by {reduction*100:.1f}% (+0.3)")
    else:
        pct = f"{reduction*100:.1f}%" if reduction is not None else "N/A"
        feedback_parts.append(f"✗ Loss reduction {pct} — need ≥50% over 30 steps")

    return round(score, 2), "\n".join(feedback_parts)
