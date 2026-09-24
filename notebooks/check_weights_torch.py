"""Check the per-example weight column of the neural loss (issues #32/#33).

Run as its own process (PyTorch and LightGBM cannot share one on macOS). Prints one JSON line.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from dcn import DCNMTL, bce_loss, train_model

torch.manual_seed(0)
model = DCNMTL(n_num=6, n_cats=5)
x = torch.randn(64, 6)
cat = torch.randint(0, 5, (64,))
y = (torch.rand(64, 3) < 0.3).float()
tw = (1.0, 2.0, 4.0)
logits = model(x, cat)
plain, _ = bce_loss(logits, y, tw)
ones, _ = bce_loss(logits, y, tw, torch.ones(64))
weights = torch.linspace(0.2, 5.0, 64)
weighted, _ = bce_loss(logits, y, tw, weights)
zero, _ = bce_loss(logits, y, tw, torch.zeros(64))

# training: weights of 1 reproduce the unweighted run exactly; non-uniform weights change the result
def run(w):
    torch.manual_seed(1)
    m = DCNMTL(n_num=6, n_cats=5)
    xs, cs = torch.randn(512, 6), torch.randint(0, 5, (512,))
    ys = (torch.rand(512, 3) < 0.3).float()
    train_model(m, xs, cs, ys, xs[:128], cs[:128], ys[:128], task_weights=tw, epochs=2, batch=128, w_tr=w, label="")
    return torch.cat([p.flatten() for p in m.parameters()]).detach().numpy()


base = run(None)
same = run(torch.ones(512))
diff = run(torch.linspace(0.1, 8.0, 512))
print("RESULT " + json.dumps({
    "loss_plain": plain.item(), "loss_ones": ones.item(), "loss_weighted": weighted.item(), "loss_zero": zero.item(),
    "train_ones_equals_plain": bool(np.allclose(base, same, atol=1e-7)),
    "train_weighted_differs": bool(not np.allclose(base, diff, atol=1e-4)),
}))
