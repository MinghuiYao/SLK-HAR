"""Dataset-free SLK-HAR forward/backward smoke example."""

import torch

from slk_har import SLKNet


def main() -> None:
    torch.manual_seed(7)
    model = SLKNet((1, 128, 9), 6)
    logits = model(torch.randn(4, 1, 128, 9))
    logits.mean().backward()
    print(f"logits: {tuple(logits.shape)}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    main()
