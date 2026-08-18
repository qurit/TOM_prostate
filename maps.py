"""Accumulation of registered cases into population-level TOM maps.

Everything is computed in a single streaming pass over the per-case NIfTI files
(sum, sum-of-squares and component extent together), so peak memory is a
handful of template-sized arrays regardless of cohort size.

Maps produced
-------------
``TOM_occurrence``      P(lesion at v) = (1/N) * sum_i M_i(v)
``TOM_suv_mean``        E[SUV | lesion at v]
``TOM_suv_cv``          voxel-wise coefficient of variation of lesion SUV
``TOM_extent_voxels``   E[connected component extent | lesion at v], in voxels
``TOM_lesion_count``    number of patients contributing a lesion at v
``TOM_volume_mean``     E[volume | lesion at v] — see the note in build_maps
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from .config import TOMConfig
from .io import image_like, write_nifti
from .template import TemplateSpace, load_template

log = logging.getLogger(__name__)

MASK_SUFFIX = "_mask_reg.nii.gz"
SUV_SUFFIX = "_suv_reg.nii.gz"
VOL_SUFFIX = "_vol_reg.nii.gz"

EPS = 1e-8


@dataclass
class TOMMaps:
    """The population maps, in template space."""

    occurrence: np.ndarray
    suv_mean: np.ndarray
    suv_cv: np.ndarray
    extent: np.ndarray
    volume_mean: np.ndarray
    lesion_count: np.ndarray
    n_cases: int
    voxel_volume_mm3: float
    paths: dict[str, Path]

    def describe(self) -> str:
        pos = self.occurrence > 0
        if not pos.any():
            return f"N={self.n_cases}: no lesion voxels"
        return (
            f"N={self.n_cases}  "
            f"P(lesion) max={self.occurrence.max():.4f} "
            f"mean(pos)={self.occurrence[pos].mean():.4f}  "
            f"SUV mean(pos)={self.suv_mean[pos].mean():.2f}  "
            f"extent mean(pos)={self.extent[pos].mean():.2f} vox"
        )


def registered_case_ids(config: TOMConfig) -> list[str]:
    return [f.name[: -len(MASK_SUFFIX)] for f in sorted(config.mask_dir.glob(f"*{MASK_SUFFIX}"))]


def build_maps(
    config: TOMConfig,
    template: TemplateSpace | None = None,
    case_ids: Iterable[str] | None = None,
    save: bool = True,
    progress: bool = True,
) -> TOMMaps:
    """Accumulate every registered case into the population maps.

    Note on ``TOM_volume_mean``: it is written for completeness but should not
    be interpreted at the 8 mm TOM grid, where a single voxel is ~2 cm³ and the
    median PSMA lesion is smaller than that. ``TOM_extent_voxels`` is the map to
    report for lesion size at this resolution.
    """
    template = template or load_template(config)
    ids = list(case_ids) if case_ids is not None else registered_case_ids(config)
    if not ids:
        raise FileNotFoundError(
            f"no registered masks in {config.mask_dir}; run `psma-tom register` first"
        )

    shape = template.shape
    voxel_volume = template.voxel_volume_mm3

    occur_sum = np.zeros(shape, dtype=np.float64)
    suv_sum = np.zeros(shape, dtype=np.float64)
    suv_sq_sum = np.zeros(shape, dtype=np.float64)
    vol_sum = np.zeros(shape, dtype=np.float64)
    ext_sum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)

    iterator: Iterable[str] = ids
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(ids, desc="accumulate")
        except ImportError:  # pragma: no cover
            pass

    n_used = 0
    for case_id in iterator:
        mask_path = config.mask_dir / f"{case_id}{MASK_SUFFIX}"
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        if mask.shape != shape:
            log.warning("%s: shape %s != template %s — skipped", case_id, mask.shape, shape)
            continue

        binary = mask > 0
        occur_sum += binary
        n_used += 1
        if not binary.any():
            continue

        count[binary] += 1.0

        suv_path = config.suv_dir / f"{case_id}{SUV_SUFFIX}"
        if suv_path.exists():
            suv = sitk.GetArrayFromImage(
                sitk.Cast(sitk.ReadImage(str(suv_path)), sitk.sitkFloat32)
            ).astype(np.float64)
            suv_sum[binary] += suv[binary]
            suv_sq_sum[binary] += suv[binary] ** 2

        vol_path = config.vol_dir / f"{case_id}{VOL_SUFFIX}"
        if vol_path.exists():
            vol = sitk.GetArrayFromImage(
                sitk.Cast(sitk.ReadImage(str(vol_path)), sitk.sitkFloat32)
            ).astype(np.float64)
            vol_sum[binary] += vol[binary]

        ext_sum += _component_extent(binary, max_voxels=config.max_lesion_voxels_tom)

    occurrence = (occur_sum / max(n_used, 1)).astype(np.float32)
    suv_mean = _safe_divide(suv_sum, count)
    volume_mean = _safe_divide(vol_sum, count)
    extent = _safe_divide(ext_sum, count)
    suv_cv = _coefficient_of_variation(suv_sum, suv_sq_sum, count)

    maps = TOMMaps(
        occurrence=occurrence,
        suv_mean=suv_mean,
        suv_cv=suv_cv,
        extent=extent,
        volume_mean=volume_mean,
        lesion_count=count.astype(np.float32),
        n_cases=n_used,
        voxel_volume_mm3=voxel_volume,
        paths={},
    )

    if save:
        maps.paths = _save_maps(config, maps, template)
    log.info(maps.describe())
    return maps


def build_extent_map(
    config: TOMConfig, template: TemplateSpace | None = None, save: bool = True
) -> np.ndarray:
    """Recompute only the disease-extent map from the registered masks."""
    template = template or load_template(config)
    shape = template.shape
    ext_sum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)

    for case_id in registered_case_ids(config):
        mask = sitk.GetArrayFromImage(
            sitk.ReadImage(str(config.mask_dir / f"{case_id}{MASK_SUFFIX}"))
        )
        binary = mask > 0
        if not binary.any():
            continue
        ext_sum += _component_extent(binary, max_voxels=config.max_lesion_voxels_tom)
        count[binary] += 1.0

    extent = _safe_divide(ext_sum, count)
    if save:
        write_nifti(
            image_like(extent, template.downsampled), config.output_dir / "TOM_extent_voxels.nii.gz"
        )
    return extent


def coverage_summary(
    config: TOMConfig,
    maps: TOMMaps | None = None,
    template: TemplateSpace | None = None,
    save: bool = True,
):
    """Fraction of the template body covered at a range of occurrence cut-offs."""
    import pandas as pd

    template = template or load_template(config)
    maps = maps or build_maps(config, template=template, save=False, progress=False)

    body = sitk.GetArrayFromImage(template.body).astype(bool)
    body_voxels = max(int(body.sum()), 1)

    rows = []
    for threshold in config.coverage_thresholds:
        selected = maps.occurrence > threshold
        n = int(selected.sum())
        rows.append(
            {
                "occurrence_threshold": threshold,
                "n_voxels": n,
                "body_pct": round(100.0 * n / body_voxels, 2),
                "mean_suv": round(float(maps.suv_mean[selected].mean()) if n else 0.0, 3),
                "mean_extent_voxels": round(float(maps.extent[selected].mean()) if n else 0.0, 2),
                "max_patients_per_voxel": int(maps.lesion_count[selected].max()) if n else 0,
            }
        )

    summary = pd.DataFrame(rows)
    if save:
        path = config.output_dir / "TOM_coverage_summary.csv"
        summary.to_csv(path, index=False)
        log.info("wrote %s", path)
    return summary


# ── internals ────────────────────────────────────────────────────────────────
def _component_extent(binary: np.ndarray, max_voxels: int | None = None) -> np.ndarray:
    """Per-voxel size (in voxels) of the connected component it belongs to."""
    labelled, n_components = ndi.label(binary)
    out = np.zeros(binary.shape, dtype=np.float64)
    if not n_components:
        return out

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background carries no extent
    if max_voxels is not None:
        sizes[1:][sizes[1:] > max_voxels] = 0  # drop confluent artefacts
    return sizes[labelled].astype(np.float64)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide only where the denominator is non-zero, leaving 0 elsewhere.

    Dividing by ``count + eps`` instead would bias every voxel slightly and, for
    voxels seen by a single patient, leave a spurious non-zero variance.
    """
    out = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out.astype(np.float32)


def _coefficient_of_variation(
    total: np.ndarray, total_sq: np.ndarray, count: np.ndarray
) -> np.ndarray:
    mean = _safe_divide(total, count).astype(np.float64)
    mean_sq = _safe_divide(total_sq, count).astype(np.float64)
    variance = np.maximum(mean_sq - mean**2, 0.0)
    return _safe_divide(np.sqrt(variance), np.where(mean > 1e-6, mean, 0.0))


def _save_maps(config: TOMConfig, maps: TOMMaps, template: TemplateSpace) -> dict[str, Path]:
    named = {
        "occurrence": ("TOM_occurrence.nii.gz", maps.occurrence),
        "suv_mean": ("TOM_suv_mean.nii.gz", maps.suv_mean),
        "suv_cv": ("TOM_suv_cv.nii.gz", maps.suv_cv),
        "extent": ("TOM_extent_voxels.nii.gz", maps.extent),
        "volume_mean": ("TOM_volume_mean.nii.gz", maps.volume_mean),
        "lesion_count": ("TOM_lesion_count.nii.gz", maps.lesion_count),
    }
    paths: dict[str, Path] = {}
    for key, (filename, array) in named.items():
        paths[key] = write_nifti(
            image_like(array, template.downsampled), config.output_dir / filename
        )
        log.info("wrote %s", paths[key])
    return paths
