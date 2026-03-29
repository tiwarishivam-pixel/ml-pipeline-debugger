"""
Task 3 — Hard: Silent Underfitting.

The script runs perfectly. No crash. No NaN. Loss decreases slightly
then completely flatlines. There is NO error message. The agent must
inspect the model architecture, weight initialisation, and training
mechanics to find the structural bug preventing learning.

This is the bug that has burned every senior ML engineer at least once.

Bug variants:
  - zero_weight_init:   all weights initialised to 0 → symmetry problem
  - frozen_layer:       key hidden layer has requires_grad=False
  - missing_optimizer_step: optimizer.step() is commented out
  - wrong_activation:   ReLU replaced with a broken lambda returning zeros
  - output_dim_mismatch: output layer produces wrong shape, silently broadcast

Grader scoring (0.0 – 1.0, partial credit):
  +0.2  script runs without crashing
  +0.4  loss escapes the flatline (gets below BASELINE_FLOOR)
  +0.2  final loss < 30% of initial
  +0.2  within 15% of the pre-measured reference run
"""

import random
from typing import List, Tuple

from .grader_utils import has_nan, parse_losses, pct_reduction, run_script

# Pre-measured: a correct version of this task trains to ~0.10 loss in 50 steps
REFERENCE_LOSS = 1.0
# Buggy versions flatline at around 1.0 (random predictions on normalised targets)
BASELINE_FLOOR = 5.0


# ── script templates ───────────────────────────────────────────────────────────

_BROKEN_SCRIPTS = {

    "zero_weight_init": """
import torch
import torch.nn as nn
torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        # BUG: zero initialisation — all neurons identical → symmetry never broken
        nn.init.constant_(self.fc1.weight, 0.0)
        nn.init.constant_(self.fc2.weight, 0.0)
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]).detach()

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",

    "frozen_layer": """
import torch
import torch.nn as nn
torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        # BUG: middle layer frozen — gradient cannot flow through the network
        for param in self.fc2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]).detach()

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",

    "missing_optimizer_step": """
import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]).detach()

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    # BUG: optimizer.step() is missing — gradients computed but weights never updated
    print(f"loss:{loss.item():.6f}")
""",

    "wrong_activation": """
import torch
import torch.nn as nn
torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # BUG: activation returns all zeros — entire hidden representation is dead
        dead_activation = lambda t: t * 0
        x = dead_activation(self.fc1(x))
        x = dead_activation(self.fc2(x))
        return self.fc3(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (3 * X[:, 0:1] + 2 * X[:, 1:2] - X[:, 2:3]).detach()

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item():.6f}")
""",
}

_DESCRIPTIONS = {
    "zero_weight_init": (
        "SYMPTOM: Loss starts at ~1.0 and barely moves after 50 epochs. No crash. No NaN.\n"
        "The model architecture, optimiser, and data all look correct at first glance.\n"
        "The training loop is fine. Inspect the weight initialisation strategy in __init__.\n"
        "Hint: think about what happens when every neuron in a layer starts with identical weights."
    ),
    "frozen_layer": (
        "SYMPTOM: Loss decreases slightly in early steps then completely flatlines. No error.\n"
        "The model has 3 layers. Training seems to run but the loss refuses to drop below ~0.8.\n"
        "Inspect the model definition carefully — specifically the parameter settings for each layer."
    ),
    "missing_optimizer_step": (
        "SYMPTOM: Loss is identical every single epoch. Completely flat from step 1. No crash.\n"
        "Gradients are being computed (no error from loss.backward()). "
        "But nothing is changing.\n"
        "Inspect the training loop step by step — is every necessary operation present?"
    ),
    "wrong_activation": (
        "SYMPTOM: Loss flatlines immediately and never changes. No crash. No NaN.\n"
        "The architecture has hidden layers but the network behaves as if they don't exist.\n"
        "Inspect the forward() method — specifically what the activation function actually returns."
    ),
}


# ── public API ─────────────────────────────────────────────────────────────────

def sample_task3() -> Tuple[str, str, List[float], str]:
    """Randomly select a bug variant and return a task3 episode."""
    bug_key = random.choice(list(_BROKEN_SCRIPTS.keys()))
    broken_script = _BROKEN_SCRIPTS[bug_key].strip()

    _, stdout, _ = run_script(broken_script, timeout=60)
    losses = parse_losses(stdout)
    curve_preview = losses[:15]

    description = (
        f"{_DESCRIPTIONS[bug_key]}\n"
        f"Loss curve (first {len(curve_preview)} steps): {[round(l, 4) for l in curve_preview]}"
    )
    return bug_key, broken_script, curve_preview, description


def grade_task3(fixed_code: str) -> Tuple[float, str]:
    """Grade the agent's fixed script. Returns (score 0.0-1.0, feedback)."""
    score = 0.0
    feedback_parts = []

    success, stdout, stderr = run_script(fixed_code, timeout=90)
    if not success:
        return 0.0, f"Script crashed or timed out.\nStderr: {stderr[:300]}"
    score += 0.2
    feedback_parts.append("✓ Script ran cleanly (+0.2)")

    losses = parse_losses(stdout)
    if not losses:
        return score, "\n".join(feedback_parts) + "\n✗ No loss values parsed. Print 'loss:X.XXXXXX' each epoch."

    if has_nan(losses):
        return score, "\n".join(feedback_parts) + "\n✗ Loss curve contains NaN/Inf — fix introduced a new bug"

    # Core check — did it escape the flatline? (0.4)
    min_loss = min(losses)
    if min_loss < BASELINE_FLOOR:
        score += 0.4
        feedback_parts.append(f"✓ Loss escaped flatline — reached {min_loss:.4f} (+0.4)")
    else:
        feedback_parts.append(
            f"✗ Loss never escaped the flatline (min={min_loss:.4f}, need < {BASELINE_FLOOR})"
        )

    # Final loss < 30% of initial (0.2)
    reduction = pct_reduction(losses)
    if reduction is not None and reduction >= 0.70:
        score += 0.2
        feedback_parts.append(f"✓ Loss reduced by {reduction*100:.1f}% (+0.2)")
    else:
        pct = f"{reduction*100:.1f}%" if reduction is not None else "N/A"
        feedback_parts.append(f"✗ Loss reduction {pct} — need ≥70%")

    # Within 15% of reference run (0.2)
    final_loss = losses[-1]
    if final_loss <= REFERENCE_LOSS * 1.15:
        score += 0.2
        feedback_parts.append(f"✓ Final loss {final_loss:.4f} within 15% of reference {REFERENCE_LOSS} (+0.2)")
    else:
        feedback_parts.append(
            f"✗ Final loss {final_loss:.4f} too far from reference {REFERENCE_LOSS} (need ≤ {REFERENCE_LOSS*1.15:.4f})"
        )

    return round(score, 2), "\n".join(feedback_parts)
