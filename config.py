"""Configuration for the multi-parametric TOM pipeline.

All tunable parameters live here so that a run can be reproduced from a single
YAML file. Nothing else in the package hard-codes a path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_CASE_COLUMNS = ("case_id", "ct_path", "mask_path", "pet_path")


@dataclass
class TOMConfig:
    """Paths and parameters for one TOM run.

    Attributes
    ----------
    case_csv:
        CSV with one row per patient and the columns listed in
        ``REQUIRED_CASE_COLUMNS``. Paths may be absolute or relative to
        ``data_root``.
    output_dir:
        Everything the pipeline writes goes under this directory.
    shrink_factor:
        Isotropic shrink applied to CT/mask/scalar maps before registration.
        With 2 mm source data a factor of 4 gives the 8 mm TOM grid.
    registration:
        ``"unigradicon"`` (three-stage: centroid translation -> deformable,
        affine fallback) or ``"affine"`` (Mattes MI, no deep model).
    """

    # ── paths ────────────────────────────────────────────────────────────────
    case_csv: Path = Path("data/cases_psma.csv")
    output_dir: Path = Path("results_psma_TOM_multi")
    data_root: Path | None = None

    # ── preprocessing ────────────────────────────────────────────────────────
    shrink_factor: int = 4
    body_hu_threshold: int = -300
    ct_clamp: tuple[int, int] = (-1024, 1500)

    # ── lesion handling ──────────────────────────────────────────────────────
    min_lesion_voxels: int = 3
    max_lesion_voxels_tom: int = 50  # confluent-artefact guard at TOM resolution

    # ── registration ─────────────────────────────────────────────────────────
    registration: str = "unigradicon"
    unigradicon_shrink: int = 2
    unigradicon_finetune_steps: int = 0
    affine_bins: int = 50
    affine_iterations: int = 200

    # ── map building / display ───────────────────────────────────────────────
    occurrence_threshold: float = 0.01
    suv_display_cap: float = 25.0
    coverage_thresholds: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20)

    # ── runtime ──────────────────────────────────────────────────────────────
    threads: int = 8
    seed: int = 42
    extras: dict[str, Any] = field(default_factory=dict)

    # ── derived directories ──────────────────────────────────────────────────
    @property
    def mask_dir(self) -> Path:
        return self.output_dir / f"{self._tag}_masks"

    @property
    def suv_dir(self) -> Path:
        return self.output_dir / f"{self._tag}_suv_maps"

    @property
    def vol_dir(self) -> Path:
        return self.output_dir / f"{self._tag}_vol_maps"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def qc_dir(self) -> Path:
        return self.output_dir / "qc"

    @property
    def _tag(self) -> str:
        return "ug" if self.registration == "unigradicon" else "affine"

    @property
    def template_ct(self) -> Path:
        return self.output_dir / "TOM_template_ct.nii.gz"

    @property
    def template_ct_ds(self) -> Path:
        return self.output_dir / "TOM_template_ct_ds.nii.gz"

    @property
    def template_body_mask(self) -> Path:
        return self.output_dir / "TOM_body_mask.nii.gz"

    # ── construction ─────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self.case_csv = Path(self.case_csv)
        self.output_dir = Path(self.output_dir)
        if self.data_root is not None:
            self.data_root = Path(self.data_root)
        if self.registration not in {"unigradicon", "affine"}:
            raise ValueError(
                f"registration must be 'unigradicon' or 'affine', got {self.registration!r}"
            )
        if self.shrink_factor < 1:
            raise ValueError("shrink_factor must be >= 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TOMConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extras = {k: v for k, v in raw.items() if k not in known}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs.setdefault("extras", {}).update(extras)
        if "ct_clamp" in kwargs:
            kwargs["ct_clamp"] = tuple(kwargs["ct_clamp"])
        if "coverage_thresholds" in kwargs:
            kwargs["coverage_thresholds"] = tuple(kwargs["coverage_thresholds"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    def ensure_dirs(self) -> TOMConfig:
        for d in (
            self.output_dir,
            self.mask_dir,
            self.suv_dir,
            self.vol_dir,
            self.figure_dir,
            self.qc_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, path: str | Path) -> Path:
        """Resolve a per-case path against ``data_root`` when it is relative."""
        p = Path(path)
        if p.is_absolute() or self.data_root is None:
            return p
        return self.data_root / p
