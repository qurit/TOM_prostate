"""Allometric scaling of tumour burden, and its voxel-wise deviation map.

Across a cohort, total lesion activity scales with metabolic tumour volume as a
power law::

    TLA = a * MTV^b

Fitting is done by ordinary least squares in log-log space. The residual of a
single patient from that law — how much more or less active their disease is
than its volume predicts — is the *deviation from the scaling law* (DSL)::

    DSL_i = log10(TLA_i) - log10(a * MTV_i^b)

Averaging DSL over the patients who have a lesion at a given voxel turns a
patient-level residual into a map: regions where disease is systematically more
avid than its bulk would predict light up positive.

The fit here is estimated from whatever cohort is passed in. Constants from a
previous cohort are never assumed — pass a fitted :class:`AllometricFit`
explicitly if you want to apply one cohort's law to another.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .config import TOMConfig
from .io import image_like, load_cases, read_image, write_nifti
from .lesions import total_metrics
from .maps import MASK_SUFFIX, _safe_divide, registered_case_ids
from .preprocessing import resample_to_geometry
from .template import TemplateSpace, load_template

log = logging.getLogger(__name__)


@dataclass
class AllometricFit:
    """A fitted power law ``TLA = a * MTV^b`` with its goodness of fit."""

    a: float
    b: float
    r2: float
    n: int
    residual_sd: float

    def predict(self, mtv: np.ndarray | float) -> np.ndarray:
        return self.a * np.power(np.asarray(mtv, dtype=float), self.b)

    def deviation(self, mtv: np.ndarray | float, tla: np.ndarray | float) -> np.ndarray:
        """Signed log10 residual from the law (the DSL)."""
        mtv = np.asarray(mtv, dtype=float)
        tla = np.asarray(tla, dtype=float)
        valid = (mtv > 0) & (tla > 0)
        out = np.full(np.broadcast(mtv, tla).shape, np.nan, dtype=float)
        out[valid] = np.log10(tla[valid]) - np.log10(self.predict(mtv[valid]))
        return out

    def describe(self) -> str:
        return (
            f"TLA = {self.a:.4f} x MTV^{self.b:.4f}  "
            f"(R2 = {self.r2:.4f}, n = {self.n}, residual SD = {self.residual_sd:.3f} log10)"
        )


def fit_power_law(mtv: Sequence[float], tla: Sequence[float]) -> AllometricFit:
    """Least-squares fit of ``TLA = a * MTV^b`` in log-log space.

    Patients with zero burden are dropped rather than clipped — a log of zero is
    not a small number, and substituting one silently anchors the intercept.
    """
    mtv = np.asarray(mtv, dtype=float)
    tla = np.asarray(tla, dtype=float)
    valid = (mtv > 0) & (tla > 0) & np.isfinite(mtv) & np.isfinite(tla)
    if valid.sum() < 3:
        raise ValueError(f"need at least 3 patients with non-zero burden, got {int(valid.sum())}")

    x = np.log10(mtv[valid])
    y = np.log10(tla[valid])
    if x.std() < 1e-6:
        # Every patient has the same burden: the slope is unidentifiable and a
        # least-squares fit would return an arbitrary exponent with R2 = 0
        # rather than failing. Better to say so than to publish the number.
        raise ValueError(
            "MTV has no spread across the cohort; the scaling exponent is unidentifiable"
        )
    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fit = AllometricFit(
        a=float(10**intercept),
        b=float(slope),
        r2=float(r2),
        n=int(valid.sum()),
        residual_sd=float(residuals.std(ddof=1)) if valid.sum() > 1 else 0.0,
    )
    log.info(fit.describe())
    return fit


def cohort_metrics(
    config: TOMConfig, df=None, case_ids: Iterable[str] | None = None, progress: bool = True
):
    """Per-patient MTV and TLA, measured at the original acquisition resolution.

    Deliberately not measured on the registered masks: resampling to the 8 mm
    template grid changes lesion volumes, and the scaling law is a property of
    the imaging, not of the template.
    """
    import pandas as pd

    df = load_cases(config) if df is None else df
    if case_ids is not None:
        wanted = {str(c) for c in case_ids}
        df = df[df["case_id"].isin(wanted)]

    rows = list(df.iterrows())
    iterator = rows
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(rows, desc="MTV / TLA")
        except ImportError:  # pragma: no cover
            pass

    records = []
    for _, row in iterator:
        case_id = str(row["case_id"])
        try:
            mask = read_image(config.resolve(row["mask_path"]))
            pet = read_image(config.resolve(row["pet_path"]))
            pet = resample_to_geometry(pet, mask, sitk.sitkLinear, 0.0)
            metrics = total_metrics(mask, pet, min_voxels=config.min_lesion_voxels)
        except Exception:
            log.exception("case %s failed", case_id)
            continue
        records.append({"case_id": case_id, **metrics})

    table = pd.DataFrame(records)
    path = config.output_dir / "cohort_burden_metrics.csv"
    table.to_csv(path, index=False)
    log.info("wrote %s (%d cases)", path, len(table))
    return table


def deviation_map(
    config: TOMConfig,
    deviations: dict[str, float],
    template: TemplateSpace | None = None,
    save: bool = True,
) -> np.ndarray:
    """Voxel-wise mean DSL over the patients with a lesion at each voxel.

    Parameters
    ----------
    deviations:
        ``case_id -> DSL`` for every patient, e.g. from
        ``dict(zip(table.case_id, fit.deviation(table.mtv_cm3, table.tla)))``.
    """
    template = template or load_template(config)
    shape = template.shape

    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    used = 0

    for case_id in registered_case_ids(config):
        value = deviations.get(case_id)
        if value is None or not np.isfinite(value):
            continue
        mask = sitk.GetArrayFromImage(
            sitk.ReadImage(str(config.mask_dir / f"{case_id}{MASK_SUFFIX}"))
        )
        binary = mask > 0
        if not binary.any():
            continue
        total[binary] += float(value)
        count[binary] += 1.0
        used += 1

    dsl = _safe_divide(total, count)
    log.info("DSL map built from %d patients", used)
    if save:
        write_nifti(image_like(dsl, template.downsampled), config.output_dir / "TOM_dsl.nii.gz")
    return dsl


def build_dsl_map(
    config: TOMConfig,
    template: TemplateSpace | None = None,
    fit: AllometricFit | None = None,
    table=None,
) -> tuple[AllometricFit, np.ndarray]:
    """Convenience path: measure burden, fit the law, project residuals to voxels."""
    template = template or load_template(config)
    table = cohort_metrics(config) if table is None else table
    fit = fit or fit_power_law(table["mtv_cm3"], table["tla"])

    residuals = fit.deviation(table["mtv_cm3"].to_numpy(), table["tla"].to_numpy())
    deviations = dict(zip(table["case_id"].astype(str), residuals, strict=True))
    return fit, deviation_map(config, deviations, template=template)
