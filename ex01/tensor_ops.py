"""Perform basic PyTorch tensor operations."""

import torch


def operations(x: "torch.Tensor", y: "torch.Tensor"):
    """Return sum, matrix product, mean, and reshaped tensor."""
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[0]
    result = (x + y, x @ y, x.mean(), x.reshape(1, -1))
    return result


def main():
    """Test tensor operations."""
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[5., 6.], [7., 8.]])
    result = operations(a, b)
    print("Sum:")
    print(result[0])
    print("Matmul:")
    print(result[1])
    print("Mean of a:", result[2].item())
    print("Reshaped a:", result[3])


if __name__ == "__main__":
    main()