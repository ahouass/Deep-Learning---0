"""Compute derivatives using PyTorch Autograd."""

import torch


def gradient_at(value: float) -> float:
    """Compute the gradient of f(x) = 3x² + 2x + 1."""
    x = torch.tensor(value, requires_grad=True)
    y = 3 * x ** 2 + 2 * x + 1
    y.backward()
    return x.grad.item()


def main():
    """Compare the autograd and analytical derivatives."""
    value = 3.0
    autograd_result = gradient_at(value)
    analytical_result = 6 * value + 2

    print(f"f'(3) computed by autograd: {autograd_result}")
    print(f"f'(3) analytical (6x+2): {analytical_result}")
    print(f"Match: {abs(autograd_result - analytical_result) < 1e-6}")


if __name__ == "__main__":
    main()
