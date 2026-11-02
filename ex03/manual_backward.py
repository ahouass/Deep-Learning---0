"""Compute derivatives using PyTorch Autograd."""

import torch


def manual_grads(x: float, w: float, b: float) -> tuple[float, float]:
    """Compute partial derivatives manually using the chain rule."""
    return 2 * (w * x + b) * x, 2 * (w * x + b)


def autograd_grads(x: float, w: float, b: float) -> tuple[float, float]:
    """Compute partial derivatives using PyTorch Autograd."""
    dw = torch.tensor(float(w), requires_grad=True)
    db = torch.tensor(float(b), requires_grad=True)

    y = (dw * x + db) ** 2
    y.backward()

    return dw.grad.item(), db.grad.item()
