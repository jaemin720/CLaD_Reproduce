"""Small deterministic test doubles for external vision-language models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FakeDecisionNCEBackend(nn.Module):
    """Mimic the two public methods used from the official package."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_image_shape: tuple[int, ...] | None = None

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.last_image_shape = tuple(images.shape)
        values = images.float()
        channel_means = values.mean(dim=(-2, -1))
        global_mean = values.mean(dim=(-3, -2, -1), keepdim=False).unsqueeze(-1)
        return torch.cat((channel_means, global_mean), dim=-1) * self.scale

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        device = self.scale.device
        rows = [
            [float(len(text)), float(text.count(" ")), float(ord(text[0])), 1.0]
            for text in texts
        ]
        return torch.tensor(rows, dtype=torch.float32, device=device) * self.scale

