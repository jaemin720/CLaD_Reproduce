"""Frozen adapter around the official DecisionNCE encoder.

The official package owns CLIP resize, crop, normalization, and text
tokenization. This adapter only standardizes LIBERO tensor layout, controls the
device, and guarantees that VLM features are produced without gradients.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


OFFICIAL_DECISIONNCE_REVISION = "ebdc585c5e6833ec3a2ba77f801b15c74d7a28f8"


@dataclass(frozen=True, slots=True)
class DecisionNCEAdapterConfig:
    """Runtime identity and device settings for the frozen VLM."""

    model_name: str = "DecisionNCE-T"
    device: str = "auto"
    source_revision: str = OFFICIAL_DECISIONNCE_REVISION
    checkpoint_sha256: str = ""

    def __post_init__(self) -> None:
        if self.model_name not in {"DecisionNCE-P", "DecisionNCE-T"}:
            raise ValueError(
                "model_name must be 'DecisionNCE-P' or 'DecisionNCE-T', "
                f"got {self.model_name!r}"
            )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class DecisionNCEAdapter(nn.Module):
    """Expose a stable, frozen interface over the official DecisionNCE model."""

    def __init__(
        self,
        backend: nn.Module,
        config: DecisionNCEAdapterConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = _resolve_device(config.device)
        self.backend = backend.to(self.device)
        self._freeze_backend()

    @classmethod
    def from_pretrained(
        cls,
        config: DecisionNCEAdapterConfig,
    ) -> DecisionNCEAdapter:
        """Load the official package lazily.

        DecisionNCE is kept as an external dependency because its code revision
        and checkpoint are part of the experiment identity.
        """

        try:
            package = importlib.import_module("DecisionNCE")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "DecisionNCE is not installed. Clone "
                "https://github.com/2toinf/DecisionNCE and install it with "
                "`pip install -e <path-to-DecisionNCE>`."
            ) from error

        load = getattr(package, "load", None)
        if load is None:
            raise RuntimeError("Installed DecisionNCE package does not expose load()")

        device = _resolve_device(config.device)
        backend = load(config.model_name, device=device)
        if not isinstance(backend, nn.Module):
            raise TypeError(
                "DecisionNCE.load() must return torch.nn.Module, "
                f"got {type(backend).__name__}"
            )
        return cls(backend=backend, config=config)

    def _freeze_backend(self) -> None:
        self.backend.requires_grad_(False)
        self.backend.eval()

    def train(self, mode: bool = True) -> DecisionNCEAdapter:
        """Keep DecisionNCE in evaluation mode even when its parent trains."""

        super().train(False)
        self._freeze_backend()
        return self

    @staticmethod
    def _to_channels_first(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B,H,W,3] or [B,3,H,W], got {tuple(images.shape)}"
            )
        if images.shape[1] == 3:
            return images.contiguous()
        if images.shape[-1] == 3:
            return images.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"images must have exactly three RGB channels, got {tuple(images.shape)}"
        )

    @staticmethod
    def _validate_features(
        features: Any,
        *,
        batch_size: int,
        modality: str,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"DecisionNCE {modality} encoder must return Tensor, "
                f"got {type(features).__name__}"
            )
        if features.ndim != 2 or features.shape[0] != batch_size:
            raise ValueError(
                f"DecisionNCE {modality} features must have shape [B,D], "
                f"got {tuple(features.shape)} for batch size {batch_size}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"DecisionNCE {modality} features contain NaN or Inf")
        return features.detach()

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB images while preserving official preprocessing."""

        channels_first = self._to_channels_first(images).to(self.device)
        encode_image = getattr(self.backend, "encode_image", None)
        if encode_image is None:
            raise RuntimeError("DecisionNCE backend does not expose encode_image()")
        features = encode_image(channels_first)
        return self._validate_features(
            features,
            batch_size=channels_first.shape[0],
            modality="image",
        )

    @torch.inference_mode()
    def encode_texts(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode one or more raw language instructions."""

        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch or any(not isinstance(text, str) or not text.strip() for text in batch):
            raise ValueError("texts must contain at least one non-empty string")

        encode_text = getattr(self.backend, "encode_text", None)
        if encode_text is None:
            raise RuntimeError("DecisionNCE backend does not expose encode_text()")
        features = encode_text(batch)
        return self._validate_features(
            features,
            batch_size=len(batch),
            modality="text",
        )

    @torch.inference_mode()
    def encode_views(
        self,
        images_by_view: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode views independently; cross-view fusion is a later model concern."""

        if not images_by_view:
            raise ValueError("images_by_view cannot be empty")
        return {
            view_name: self.encode_images(images)
            for view_name, images in images_by_view.items()
        }

