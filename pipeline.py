"""Per-case registration loop.

Every case writes its three registered maps to disk as soon as they exist, and
the loop skips any case whose outputs are already present. A run that dies at
case 300 of 472 — a hosted-runtime disconnect, an OOM, a full disk — resumes by
being started again, with no bookkeeping file to keep in sync.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import SimpleITK as sitk

from .config import TOMConfig
from .io import load_cases, read_image, write_nifti
from .lesions import make_scalar_maps
from .preprocessing import downsample, normalise_ct, resample_to_geometry
from .registration import Registrar, get_registrar
from .template import TemplateSpace, load_template

log = logging.getLogger(__name__)


@dataclass
class CohortReport:
    """What happened across the cohort — goes straight into the methods section."""

    registered: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    methods: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def n_available(self) -> int:
        return self.registered + self.skipped

    def summary(self) -> str:
        parts = [
            f"registered={self.registered}",
            f"skipped(checkpoint)={self.skipped}",
            f"failed={len(self.failed)}",
        ]
        if self.methods:
            parts.append(
                "methods=" + ", ".join(f"{k}:{v}" for k, v in sorted(self.methods.items()))
            )
        return "  ".join(parts)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def case_outputs(config: TOMConfig, case_id: str) -> dict[str, Path]:
    return {
        "mask": config.mask_dir / f"{case_id}_mask_reg.nii.gz",
        "suv": config.suv_dir / f"{case_id}_suv_reg.nii.gz",
        "vol": config.vol_dir / f"{case_id}_vol_reg.nii.gz",
    }


def is_done(config: TOMConfig, case_id: str) -> bool:
    return all(p.exists() for p in case_outputs(config, case_id).values())


def register_case(
    config: TOMConfig,
    row,
    template: TemplateSpace,
    registrar: Registrar,
) -> str:
    """Register one patient and write mask, SUV and volume maps to disk.

    Returns the registration method that was used, so the caller can count how
    many cases needed the affine fallback.
    """
    case_id = str(row["case_id"])
    paths = case_outputs(config, case_id)

    ct = read_image(config.resolve(row["ct_path"]))
    mask = read_image(config.resolve(row["mask_path"]))
    pet = read_image(config.resolve(row["pet_path"]))

    # Mask and PET onto the CT grid before anything else is computed.
    mask = resample_to_geometry(mask, ct, sitk.sitkNearestNeighbor, 0.0)
    pet = resample_to_geometry(pet, ct, sitk.sitkLinear, 0.0)

    # Lesion statistics are computed at full resolution, then downsampled —
    # doing it the other way round would measure the downsampled blobs.
    scalars = make_scalar_maps(mask, pet, min_voxels=config.min_lesion_voxels)

    ct_ds = downsample(normalise_ct(ct, config.ct_clamp), config.shrink_factor)
    mask_ds = downsample(mask, config.shrink_factor)
    suv_ds = downsample(scalars.suv, config.shrink_factor)
    vol_ds = downsample(scalars.volume, config.shrink_factor)

    alignment = registrar.fit(template.downsampled, ct_ds)

    warped = {
        "mask": registrar.apply(
            alignment, mask_ds, template.downsampled, "nearest", 0.0, sitk.sitkUInt8
        ),
        "suv": registrar.apply(
            alignment, suv_ds, template.downsampled, "linear", 0.0, sitk.sitkFloat32
        ),
        "vol": registrar.apply(
            alignment, vol_ds, template.downsampled, "linear", 0.0, sitk.sitkFloat32
        ),
    }
    for key, img in warped.items():
        write_nifti(img, paths[key])

    return alignment.method


def register_cohort(
    config: TOMConfig,
    df=None,
    template: TemplateSpace | None = None,
    registrar: Registrar | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> CohortReport:
    """Register every case in the table, resuming from whatever is on disk."""
    config.ensure_dirs()
    df = load_cases(config) if df is None else df
    if limit is not None:
        df = df.head(limit)
    template = template or load_template(config)
    registrar = registrar or get_registrar(config)

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(config.threads)

    rows = list(df.iterrows())
    iterator = rows
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(rows, desc=f"register [{registrar.name}]")
        except ImportError:  # pragma: no cover
            pass

    report = CohortReport()
    methods: Counter = Counter()
    started = time.time()

    for _, row in iterator:
        case_id = str(row["case_id"])
        if is_done(config, case_id):
            report.skipped += 1
            continue
        try:
            method = register_case(config, row, template, registrar)
            methods[method] += 1
            report.registered += 1
        except Exception:
            log.exception("case %s failed", case_id)
            report.failed.append(case_id)

    report.methods = dict(methods)
    report.seconds = round(time.time() - started, 1)
    report.save(config.output_dir / "registration_report.json")
    log.info(report.summary())
    return report
