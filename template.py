"""Selection and construction of the common template space."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import TOMConfig
from .io import load_cases, read_header, read_image, write_nifti
from .preprocessing import body_mask, downsample, normalise_ct

log = logging.getLogger(__name__)


@dataclass
class TemplateSpace:
    """The reference frame every patient is mapped into.

    ``downsampled`` is the grid the TOM maps live on; ``full`` is kept so that
    maps can be projected back to a diagnostic-resolution CT for display, and
    so collaborators receiving a TOM NIfTI also get the CT it aligns to.
    """

    full: sitk.Image
    downsampled: sitk.Image
    body: sitk.Image
    case_id: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return sitk.GetArrayFromImage(self.downsampled).shape

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(self.downsampled.GetSpacing())

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    def describe(self) -> str:
        sp = [round(s, 2) for s in self.spacing]
        return (
            f"template case={self.case_id} "
            f"grid={self.downsampled.GetSize()} spacing={sp} mm "
            f"voxel={self.voxel_volume_mm3 / 1000:.2f} cm3"
        )


def select_template_case(config: TOMConfig, df=None) -> tuple[int, str]:
    """Pick the case whose voxel count is closest to the cohort median.

    Only headers are read, so this is seconds rather than the tens of minutes a
    full load of several hundred whole-body CTs would take.
    """
    df = load_cases(config) if df is None else df
    counts = []
    for _, row in df.iterrows():
        size, _ = read_header(config.resolve(row["ct_path"]))
        counts.append(float(np.prod(size)))

    counts_arr = np.asarray(counts)
    index = int(np.argmin(np.abs(counts_arr - np.median(counts_arr))))
    case_id = str(df.iloc[index]["case_id"])
    log.info("template case %s (index %d of %d)", case_id, index, len(df))
    return index, case_id


def build_template(config: TOMConfig, df=None, case_id: str | None = None) -> TemplateSpace:
    """Select (or take) a template case and write the template space to disk."""
    df = load_cases(config) if df is None else df
    config.ensure_dirs()

    if case_id is None:
        index, case_id = select_template_case(config, df)
        row = df.iloc[index]
    else:
        matches = df[df["case_id"] == str(case_id)]
        if matches.empty:
            raise KeyError(f"case_id {case_id!r} not found in {config.case_csv}")
        row = matches.iloc[0]

    full = read_image(config.resolve(row["ct_path"]))
    downsampled = downsample(normalise_ct(full, config.ct_clamp), config.shrink_factor)
    body = body_mask(downsampled, hu_threshold=config.body_hu_threshold)

    write_nifti(full, config.template_ct)
    write_nifti(downsampled, config.template_ct_ds)
    write_nifti(body, config.template_body_mask)
    (config.output_dir / "TOM_template_case_id.txt").write_text(str(case_id), encoding="utf-8")

    template = TemplateSpace(full, downsampled, body, case_id=str(case_id))
    log.info(template.describe())
    return template


def load_template(config: TOMConfig) -> TemplateSpace:
    """Reload a previously built template space (cheap; survives a disconnect)."""
    missing = [
        p
        for p in (config.template_ct, config.template_ct_ds, config.template_body_mask)
        if not Path(p).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "template not built yet; run `psma-tom template` first. Missing: "
            + ", ".join(str(m) for m in missing)
        )

    case_file = config.output_dir / "TOM_template_case_id.txt"
    case_id = case_file.read_text(encoding="utf-8").strip() if case_file.exists() else None

    return TemplateSpace(
        full=sitk.ReadImage(str(config.template_ct)),
        downsampled=sitk.ReadImage(str(config.template_ct_ds)),
        body=sitk.ReadImage(str(config.template_body_mask)),
        case_id=case_id,
    )
