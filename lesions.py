"""Per-lesion scalar maps derived from a binary segmentation.

A "scalar map" is an image the same size as the mask in which every voxel of a
lesion carries a single lesion-level number (mean SUV, volume, extent). Warping
these through the same transform as the mask keeps the lesion-level statistic
attached to its anatomical location, which is what makes the population
averages in :mod:`psma_tom.maps` interpretable.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from .io import image_like


def voxel_volume_mm3(img: sitk.Image) -> float:
    """Physical volume of a single voxel in mm³."""
    return float(np.prod(img.GetSpacing()))


class ScalarMaps(NamedTuple):
    suv: sitk.Image
    volume: sitk.Image
    n_lesions: int


def make_scalar_maps(mask: sitk.Image, pet: sitk.Image, min_voxels: int = 3) -> ScalarMaps:
    """Build per-lesion mean-SUV and volume maps at the mask's resolution.

    Parameters
    ----------
    mask:
        Binary lesion segmentation.
    pet:
        SUV image already resampled onto the mask grid.
    min_voxels:
        Connected components smaller than this are dropped as segmentation
        noise rather than counted as lesions.

    Returns
    -------
    ScalarMaps
        ``suv`` holds each lesion's mean SUV at every voxel of that lesion,
        ``volume`` holds its physical volume in mm³, and ``n_lesions`` is the
        number of components that survived the size filter.
    """
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.int32)
    pet_arr = sitk.GetArrayFromImage(sitk.Cast(pet, sitk.sitkFloat32)).astype(np.float32)
    if mask_arr.shape != pet_arr.shape:
        raise ValueError(
            f"mask shape {mask_arr.shape} != PET shape {pet_arr.shape}; "
            "resample PET onto the mask grid first"
        )

    labelled, n_components = ndi.label(mask_arr > 0)
    suv_arr = np.zeros(mask_arr.shape, dtype=np.float32)
    vol_arr = np.zeros(mask_arr.shape, dtype=np.float32)
    vox_vol = voxel_volume_mm3(mask)

    kept = 0
    if n_components:
        objects = ndi.find_objects(labelled)
        for label, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            local = labelled[sl] == label
            n_vox = int(local.sum())
            if n_vox < min_voxels:
                continue
            suv_arr[sl][local] = float(pet_arr[sl][local].mean())
            vol_arr[sl][local] = n_vox * vox_vol
            kept += 1

    return ScalarMaps(image_like(suv_arr, mask), image_like(vol_arr, mask), kept)


def extent_map(mask: sitk.Image, max_voxels: int | None = None) -> sitk.Image:
    """Map where each lesion voxel carries its component size **in voxels**.

    At the 8 mm TOM grid a physical volume is misleading — half of real PSMA
    lesions are smaller than one voxel, so a "mean volume" mostly measures the
    voxel size. Component extent in voxels is the honest quantity at this
    resolution and is what the published third TOM map reports.
    """
    arr = sitk.GetArrayFromImage(mask).astype(np.int32)
    labelled, n_components = ndi.label(arr > 0)
    out = np.zeros(arr.shape, dtype=np.float32)

    if n_components:
        for label, sl in enumerate(ndi.find_objects(labelled), start=1):
            if sl is None:
                continue
            local = labelled[sl] == label
            n_vox = int(local.sum())
            if max_voxels is not None and n_vox > max_voxels:
                continue  # confluent disease / registration artefact
            out[sl][local] = n_vox

    return image_like(out, mask)


def lesion_sizes_cm3(mask: sitk.Image, min_voxels: int = 1) -> np.ndarray:
    """Volumes (cm³) of every connected component in a mask."""
    arr = sitk.GetArrayFromImage(mask).astype(np.int32)
    labelled, n_components = ndi.label(arr > 0)
    if not n_components:
        return np.empty(0, dtype=float)

    counts = np.bincount(labelled.ravel())[1:]
    counts = counts[counts >= min_voxels]
    return counts * voxel_volume_mm3(mask) / 1000.0


def total_metrics(mask: sitk.Image, pet: sitk.Image, min_voxels: int = 3) -> dict:
    """Patient-level MTV (cm³) and TLA — the inputs to the allometric fit."""
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    pet_arr = sitk.GetArrayFromImage(sitk.Cast(pet, sitk.sitkFloat32)).astype(np.float64)
    labelled, n_components = ndi.label(mask_arr)

    mtv_mm3 = 0.0
    tla = 0.0
    vox_vol = voxel_volume_mm3(mask)
    for label in range(1, n_components + 1):
        sel = labelled == label
        n_vox = int(sel.sum())
        if n_vox < min_voxels:
            continue
        lesion_vol = n_vox * vox_vol
        mtv_mm3 += lesion_vol
        tla += float(pet_arr[sel].mean()) * lesion_vol / 1000.0

    return {
        "mtv_cm3": mtv_mm3 / 1000.0,
        "tla": tla,
        "n_lesions": int(n_components),
    }
