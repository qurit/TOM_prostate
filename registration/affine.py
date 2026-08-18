"""Affine registration with Mattes mutual information.

Used both as a standalone backend (fast, no GPU, no model download) and as the
fallback path inside the deformable backend.
"""

from __future__ import annotations

import logging

import SimpleITK as sitk

from ..preprocessing import body_mask
from .base import INTERPOLATORS, Alignment, Registrar

log = logging.getLogger(__name__)


class AffineRegistrar(Registrar):
    name = "affine"

    def __init__(
        self,
        bins: int = 50,
        iterations: int = 200,
        sampling_percentage: float = 0.20,
        use_body_masks: bool = True,
    ) -> None:
        self.bins = bins
        self.iterations = iterations
        self.sampling_percentage = sampling_percentage
        self.use_body_masks = use_body_masks

    @staticmethod
    def pyramid(fixed: sitk.Image, min_size: int = 16) -> tuple[list[int], list[float]]:
        """Multi-resolution schedule trimmed to the image actually being used.

        A fixed [4, 2, 1] pyramid is right for a full-resolution CT but
        degenerate once the coarsest level drops below a handful of voxels,
        where the optimizer will happily walk the image out of the frame.
        """
        smallest = min(fixed.GetSize())
        shrinks = [f for f in (4, 2, 1) if smallest // f >= min_size] or [1]
        sigmas = [{4: 4.0, 2: 2.0, 1: 0.0}[f] for f in shrinks]
        return shrinks, sigmas

    def fit(self, fixed: sitk.Image, moving: sitk.Image) -> Alignment:
        initial = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(self.bins)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(self.sampling_percentage)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=self.iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=15,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        shrinks, sigmas = self.pyramid(fixed)
        reg.SetShrinkFactorsPerLevel(shrinks)
        reg.SetSmoothingSigmasPerLevel(sigmas)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(initial, inPlace=False)

        if self.use_body_masks:
            # Restricting the metric to the body keeps air and table out of the
            # MI estimate. Not all inputs support it, so failure is non-fatal.
            try:
                reg.SetMetricFixedMask(body_mask(fixed))
                reg.SetMetricMovingMask(body_mask(moving))
            except RuntimeError as exc:  # pragma: no cover - backend dependent
                log.debug("body-mask metric restriction unavailable: %s", exc)

        transform = reg.Execute(fixed, moving)
        return Alignment(
            method="affine",
            transform=transform,
            metadata={
                "final_metric": reg.GetMetricValue(),
                "iterations": reg.GetOptimizerIteration(),
                "stop_condition": reg.GetOptimizerStopConditionDescription(),
            },
        )

    def apply(
        self,
        alignment: Alignment,
        image: sitk.Image,
        reference: sitk.Image,
        interp: str = "nearest",
        default_value: float = 0.0,
        out_type: int | None = None,
    ) -> sitk.Image:
        out = sitk.Resample(
            image,
            reference,
            alignment.transform,
            INTERPOLATORS[interp],
            default_value,
            image.GetPixelID(),
        )
        return sitk.Cast(out, out_type) if out_type is not None else out
