"""Introduction to PyTorch tensors."""

import torch


def describe(x: "torch.Tensor") -> None:
    """Print the tensor's main properties."""
    print("Values:")
    print(x)
    print("Shape:", x.shape)
    print("Dimensions:", x.ndim)
    print("Dtype:", x.dtype)
    print("Device:", x.device)


def main():
    """Test the describe function."""
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    describe(x)


if __name__ == "__main__":
    main()
