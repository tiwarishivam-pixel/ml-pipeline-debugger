"""
Task 1 — Easy: Hyperparameter Bomb.

The agent receives a small PyTorch training script where exactly ONE
hyperparameter has been deliberately broken. The script runs without crashing
but learns terribly. The agent must identify and fix the bad value.

Bug variants (one randomly selected per episode):
  - lr_too_high:   lr=10.0          → loss explodes
  - lr_zero:       lr=0.0           → loss never moves
  - weight_decay:  weight_decay=500 → weights killed immediately
  - momentum_neg:  momentum=-0.9    → gradient updates fight themselves
  - epochs_zero:   epochs=0         → no training at all

Grader scoring (0.0 – 1.0, partial credit):
  +0.3  script runs without crashing
  +0.4  loss is decreasing (negative slope over last 5 steps)
  +0.3  final loss < 10% of initial loss  (90% reduction)
"""

import random
from typing import Dict, Tuple

from .grader_utils import (
    has_nan,
    is_decreasing,
    parse_losses,
    pct_reduction,
    run_script,
)

# ── bug catalogue ──────────────────────────────────────────────────────────────

_BUGS: Dict[str, Dict] = {
    "lr_too_high": {
        "param": "lr=10.0",
        "correct": "lr=0.01",
        "symptom": "Loss is exploding — increases every step.",
        "hint_line": "optimizer = torch.optim.SGD(model.parameters(), lr=10.0)",
    },
    "lr_zero": {
        "param": "lr=0.0",
        "correct": "lr=0.01",
        "symptom": "Loss is completely flat — never decreases at all.",
        "hint_line": "optimizer = torch.optim.SGD(model.parameters(), lr=0.0)",
    },
    "weight_decay_too_high": {
        "param": "weight_decay=500.0",
        "correct": "weight_decay=1e-4",
        "symptom": "Loss drops to near-zero instantly then flatlines — weights are being killed by L2 penalty.",
        "hint_line": "optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=500.0)",
    },
    "momentum_negative": {
        "param": "momentum=-0.9",
        "correct": "momentum=0.9",
        "symptom": "Loss oscillates wildly and diverges — gradient updates are fighting each other.",
        "hint_line": "optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=-0.9)",
    },
    "epochs_zero": {
        "param": "range(0)",
        "correct": "range(30)",
        "symptom": "No loss values printed at all — training loop never executes.",
        "hint_line": "for epoch in range(0):",
    },
}

# ── script template ────────────────────────────────────────────────────────────

def _make_broken_script(bug_key: str) -> str:
    """
    Build a complete broken training script for the given bug variant.
    The bug is clearly present in the code — the agent must spot and fix it.
    """
    bug = _BUGS[bug_key]

    scripts = {
        "lr_too_high": f"""
import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
criterion = nn.MSELoss()

X = torch.randn(200, 10)
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]

for epoch in range(30):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{{loss.item():.6f}}")
""",
        "lr_zero": f"""
import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
criterion = nn.MSELoss()

X = torch.randn(200, 10)
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]

for epoch in range(30):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{{loss.item():.6f}}")
""",
        "weight_decay_too_high": f"""
import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=500.0)
criterion = nn.MSELoss()

X = torch.randn(200, 10)
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]

for epoch in range(30):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{{loss.item():.6f}}")
""",
        "momentum_negative": f"""
import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=-0.9)
criterion = nn.MSELoss()

X = torch.randn(200, 10)
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]

for epoch in range(30):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{{loss.item():.6f}}")
""",
        "epochs_zero": f"""
import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

X = torch.randn(200, 10)
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]

for epoch in range(0):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{{loss.item():.6f}}")
""",
    }
    return scripts[bug_key].strip()


# ── public API ─────────────────────────────────────────────────────────────────

def sample_task1() -> Tuple[str, str, str, str]:
    """
    Randomly select a bug variant and return a task1 episode.

    Returns:
        (bug_key, broken_script, loss_curve_repr, task_description)
    """
    bug_key = random.choice(list(_BUGS.keys()))
    bug = _BUGS[bug_key]
    broken_script = _make_broken_script(bug_key)

    # Run the broken script to get the actual loss curve to show the agent
    _, stdout, stderr = run_script(broken_script, timeout=30)
    losses = parse_losses(stdout)

    # Format loss curve as a short list (max 10 values)
    curve_preview = losses[:10] if losses else []

    description = (
        f"SYMPTOM: {bug['symptom']}\n"
        f"Loss curve (first {len(curve_preview)} steps): {[round(l, 4) for l in curve_preview]}\n"
        f"The training script runs without crashing. Inspect the optimiser and training loop "
        f"configuration. Find and fix the single broken hyperparameter."
    )

    return bug_key, broken_script, curve_preview, description


def grade_task1(fixed_code: str) -> Tuple[float, str]:
    """
    Grade the agent's fixed script.

    Returns:
        (score: float 0.0-1.0, feedback: str)
    """
    score = 0.0
    feedback_parts = []

    # ── Step 1: does it run? (0.3) ────────────────────────────────────────────
    success, stdout, stderr = run_script(fixed_code, timeout=45)
    if not success:
        return 0.0, f"Script crashed or timed out.\nStderr: {stderr[:300]}"
    score += 0.3
    feedback_parts.append("✓ Script ran cleanly (+0.3)")

    losses = parse_losses(stdout)
    if not losses:
        return score, "\n".join(feedback_parts) + "\n✗ No loss values parsed from stdout. Print loss as 'loss:X.XXXXXX'"

    # ── Step 2: is loss decreasing? (0.4) ────────────────────────────────────
    if is_decreasing(losses, window=5):
        score += 0.4
        feedback_parts.append("✓ Loss is decreasing (+0.4)")
    else:
        feedback_parts.append(f"✗ Loss is not decreasing. Curve: {[round(l,4) for l in losses[:8]]}")

    # ── Step 3: 90% reduction from start to end (0.3) ────────────────────────
    reduction = pct_reduction(losses)
    if reduction is not None and reduction >= 0.90:
        score += 0.3
        feedback_parts.append(f"✓ Loss reduced by {reduction*100:.1f}% (+0.3)")
    else:
        pct = f"{reduction*100:.1f}%" if reduction is not None else "N/A"
        feedback_parts.append(f"✗ Loss reduction {pct} — need ≥90% reduction")

    return round(score, 2), "\n".join(feedback_parts)
