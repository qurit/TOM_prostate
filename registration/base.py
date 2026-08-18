"""Common interface shared by every registration backend."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import SimpleITK as sitk

INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
}


@dataclass
class Alignment:
    """Result of registering one patient to the template.

    ``method`` records what actually ran, which matters because the
    deformable backend silently falls back to affine when the field of view
    leaves no overlap. Reporting how many cases fell back belongs in the paper.
    """

    method: str
    transform: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Registrar(abc.ABC):
    """Registers a moving image to a fixed template and warps images with it."""

    name: str = "base"

    @abc.abstractmethod
    def fit(self, fixed: sitk.Image, moving: sitk.Image) -> Alignment:
        """Estimate the transform mapping ``moving`` into ``fixed`` space."""

    @abc.abstractmethod
    def apply(
        self,
        alignment: Alignment,
        image: sitk.Image,
        reference: sitk.Image,
        interp: str = "nearest",
        default_value: float = 0.0,
        out_type: int | None = None,
    ) -> sitk.Image:
        """Resample ``image`` into ``reference`` space using ``alignment``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
