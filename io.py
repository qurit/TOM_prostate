"""Image and case-table I/O."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import REQUIRED_CASE_COLUMNS

log = logging.getLogger(__name__)


def read_image(path: str | Path) -> sitk.Image:
    """Read a NIfTI file or a DICOM series directory."""
    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(p))
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found in {p}")
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(p), series_ids[0]))
        return reader.Execute()
    if not p.exists():
        raise FileNotFoundError(p)
    return sitk.ReadImage(str(p))


def read_header(path: str | Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Read size and spacing without loading pixel data.

    Scanning a few hundred whole-body CTs this way takes seconds instead of
    tens of minutes, which is what makes template selection cheap.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return tuple(reader.GetSize()), tuple(reader.GetSpacing())


def write_nifti(img: sitk.Image, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))
    return p


def image_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Wrap a numpy array as an image sharing ``reference``'s geometry."""
    img = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    img.CopyInformation(reference)
    return img


def as_array(img: sitk.Image, dtype=np.float64) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.Cast(img, sitk.sitkFloat32)).astype(dtype)


def load_cases(config) -> pd.DataFrame:  # noqa: F821 - pandas imported lazily
    """Load and validate the case table.

    Raises a clear error when the ``pet_path`` column is missing, which is the
    single most common setup mistake — the CT/mask columns come from the
    segmentation export while PET has to be derived separately.
    """
    import pandas as pd

    df = pd.read_csv(config.case_csv)
    missing = set(REQUIRED_CASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{config.case_csv} is missing column(s): {sorted(missing)}. "
            f"Expected: {list(REQUIRED_CASE_COLUMNS)}"
        )
    df["case_id"] = df["case_id"].astype(str)
    return df


def check_cases_exist(config, df, limit: int | None = None) -> list[str]:
    """Return case_ids with at least one unreadable path."""
    bad: list[str] = []
    rows = df if limit is None else df.head(limit)
    for _, row in rows.iterrows():
        for col in ("ct_path", "mask_path", "pet_path"):
            if not config.resolve(row[col]).exists():
                bad.append(str(row["case_id"]))
                break
    return bad


def derive_pet_paths(
    df, ct_column: str = "ct_path", pattern: tuple[str, str] = ("_0000.nii.gz", "_0001.nii.gz")
):
    """Add a ``pet_path`` column using the nnU-Net channel naming convention."""
    df = df.copy()
    df["pet_path"] = df[ct_column].str.replace(pattern[0], pattern[1], regex=False)
    return df


def iter_case_ids(directory: Path, suffix: str) -> Iterable[str]:
    for f in sorted(Path(directory).glob(f"*{suffix}")):
        yield f.name[: -len(suffix)]
