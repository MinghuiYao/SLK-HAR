"""Reference SLK-HAR classifier."""

from collections.abc import Sequence

from torch import Tensor, nn

from .layers import SLKBlock


class SLKNet(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        num_classes: int,
        channels: tuple[int, int, int] = (64, 128, 256),
        large_kernel: tuple[int, int] = (57, 1),
        small_kernel: tuple[int, int] = (5, 1),
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        if len(input_shape) < 2 or num_classes <= 0:
            raise ValueError("input_shape and num_classes are invalid.")
        self.num_modalities = int(input_shape[-1])
        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            SLKBlock(
                channels[0],
                channels[1],
                large_kernel,
                small_kernel,
                drop_path=drop_path,
            ),
            nn.ReLU(inplace=True),
            SLKBlock(
                channels[1],
                channels[2],
                large_kernel,
                small_kernel,
                drop_path=drop_path,
            ),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, self.num_modalities))
        self.classifier = nn.Linear(channels[-1] * self.num_modalities, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("SLKNet expects [batch, 1, time, modalities].")
        if inputs.shape[-1] != self.num_modalities:
            raise ValueError(
                f"Model configured for {self.num_modalities} modalities, got {inputs.shape[-1]}."
            )
        features = self.pool(self.features(inputs)).flatten(start_dim=1)
        return self.classifier(features)


SLK_Net = SLKNet

__all__ = ["SLKNet", "SLK_Net"]
