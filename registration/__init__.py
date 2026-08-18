"""Registration backends."""

from __future__ import annotations

from .affine import AffineRegistrar
from .base import Alignment, Registrar
from .unigradicon import UniGradICONRegistrar

__all__ = [
    "Alignment",
    "Registrar",
    "AffineRegistrar",
    "UniGradICONRegistrar",
    "get_registrar",
]


def get_registrar(config) -> Registrar:
    """Build the backend named by ``config.registration``."""
    affine = AffineRegistrar(bins=config.affine_bins, iterations=config.affine_iterations)
    if config.registration == "affine":
        return affine
    if config.registration == "unigradicon":
        return UniGradICONRegistrar(
            shrink=config.unigradicon_shrink,
            finetune_steps=config.unigradicon_finetune_steps,
            hu_threshold=config.body_hu_threshold,
            fallback=affine,
        )
    raise ValueError(f"unknown registration backend: {config.registration!r}")
