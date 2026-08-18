"""CT preprocessing: geometry harmonisation, body masking, intensity handling."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

DEFAULT_HU_THRESHOLD = -300
DEFAULT_CT_CLAMP = (-1024, 1500)


def resample_to_geometry(
    source: sitk.Image,
    reference: sitk.Image,
    interpolator: int = sitk.sitkNearestNeighbor,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample ``source`` onto ``reference``'s grid with an identity transform.

    Used to bring segmentation masks and PET onto the CT grid; PSMA PET/CT
    series are spatially aligned but rarely share a voxel grid.
    """
    return sitk.Resample(
        source, reference, sitk.Transform(), interpolator, default_value, source.GetPixelID()
    )


def body_mask(
    ct: sitk.Image, hu_threshold: int = DEFAULT_HU_THRESHOLD, closing_radius: int = 3
) -> sitk.Image:
    """Largest connected above-threshold component, hole-filled and closed."""
    binary = sitk.Greater(sitk.Cast(ct, sitk.sitkFloat32), float(hu_threshold))
    binary = sitk.BinaryFillhole(binary, foregroundValue=1)
    components = sitk.ConnectedComponent(binary)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    labels = stats.GetLabels()
    if not labels:
        return sitk.Cast(binary, sitk.sitkUInt8)

    largest = max(labels, key=stats.GetPhysicalSize)
    mask = sitk.BinaryThreshold(
        components, lowerThreshold=largest, upperThreshold=largest, insideValue=1, outsideValue=0
    )
    return sitk.BinaryMorphologicalClosing(sitk.Cast(mask, sitk.sitkUInt8), [closing_radius] * 3)


def body_centroid(ct: sitk.Image, hu_threshold: int = DEFAULT_HU_THRESHOLD) -> np.ndarray:
    """Physical centroid (mm) of the body, used to initialise registration.

    Falls back to the image centre when no body component is found, so an
    unusual field of view degrades rather than crashes.
    """
    binary = sitk.Greater(sitk.Cast(ct, sitk.sitkFloat32), float(hu_threshold))
    binary = sitk.BinaryFillhole(binary, foregroundValue=1)
    components = sitk.ConnectedComponent(binary)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)
    labels = stats.GetLabels()
    if not labels:
        centre_index = [s // 2 for s in ct.GetSize()]
        return np.asarray(ct.TransformIndexToPhysicalPoint(centre_index), dtype=float)

    largest = max(labels, key=stats.GetPhysicalSize)
    return np.asarray(stats.GetCentroid(largest), dtype=float)


def normalise_ct(ct: sitk.Image, clamp: tuple[int, int] = DEFAULT_CT_CLAMP) -> sitk.Image:
    """Clamp to a diagnostic HU window, then z-score."""
    lo, hi = clamp
    clamped = sitk.Cast(sitk.Clamp(ct, lowerBound=lo, upperBound=hi), sitk.sitkFloat32)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(clamped)
    sigma = stats.GetSigma()
    if sigma <= 1e-8:
        return clamped
    return (clamped - stats.GetMean()) / sigma


def downsample(img: sitk.Image, factor: int) -> sitk.Image:
    """Integer shrink, preserving physical spacing metadata."""
    if factor == 1:
        return img
    return sitk.Shrink(img, [factor] * img.GetDimension())


def to_unit_range(ct: sitk.Image, window: tuple[int, int] = (-1000, 1000)) -> sitk.Image:
    """Map a HU window to [0, 1] — the intensity convention uniGradICON expects."""
    lo, hi = window
    clamped = sitk.Cast(sitk.Clamp(ct, lowerBound=lo, upperBound=hi), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(clamped).astype(np.float32)
    arr = (arr - lo) / float(hi - lo)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct)
    return out
