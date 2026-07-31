from __future__ import annotations

import pytest
import torch

from clad.models import DecisionNCEAdapter, DecisionNCEAdapterConfig
from tests.fakes import FakeDecisionNCEBackend


def _adapter() -> tuple[DecisionNCEAdapter, FakeDecisionNCEBackend]:
    backend = FakeDecisionNCEBackend()
    adapter = DecisionNCEAdapter(
        backend=backend,
        config=DecisionNCEAdapterConfig(device="cpu"),
    )
    return adapter, backend


def test_adapter_freezes_backend_and_stays_in_eval_mode() -> None:
    adapter, backend = _adapter()

    adapter.train()

    assert not adapter.training
    assert not backend.training
    assert all(not parameter.requires_grad for parameter in backend.parameters())


def test_adapter_accepts_channels_last_and_channels_first_images() -> None:
    adapter, backend = _adapter()
    channels_last = torch.arange(2 * 5 * 4 * 3, dtype=torch.uint8).reshape(2, 5, 4, 3)

    features_last = adapter.encode_images(channels_last)
    features_first = adapter.encode_images(channels_last.permute(0, 3, 1, 2))

    assert backend.last_image_shape == (2, 3, 5, 4)
    assert features_last.shape == (2, 4)
    assert not features_last.requires_grad
    torch.testing.assert_close(features_last, features_first)


def test_adapter_encodes_text_and_independent_views() -> None:
    adapter, _ = _adapter()
    images = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)

    text_features = adapter.encode_texts(["pick object", "place object"])
    view_features = adapter.encode_views({"agentview_rgb": images, "wrist": images + 1})

    assert text_features.shape == (2, 4)
    assert set(view_features) == {"agentview_rgb", "wrist"}
    assert view_features["agentview_rgb"].shape == (2, 4)
    assert not torch.equal(
        view_features["agentview_rgb"],
        view_features["wrist"],
    )


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(3, 4, 4),
        torch.zeros(2, 4, 4, 2),
    ],
)
def test_adapter_rejects_invalid_image_shapes(images: torch.Tensor) -> None:
    adapter, _ = _adapter()

    with pytest.raises(ValueError, match="images must"):
        adapter.encode_images(images)

