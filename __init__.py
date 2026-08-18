"""psma_tom — population-based multi-parametric Tumor Occurrence Maps (TOM)
for PSMA PET/CT.

The package builds four population-level maps in a shared template space:

    occurrence   P(lesion at voxel v)
    suv          E[SUV | lesion at v]
    extent       E[connected lesion extent | lesion at v]
    dsl          E[deviation from the allometric scaling law | lesion at v]

plus a voxel-wise coefficient of variation of SUV and a patient-count map.

Typical use::

    from psma_tom import TOMConfig, build_template, register_cohort, build_maps

    cfg = TOMConfig.from_yaml("configs/default.yaml")
    tmpl = build_template(cfg)
    register_cohort(cfg, template=tmpl)
    build_maps(cfg)
"""

from importlib.metadata import PackageNotFoundError, version

from .allometry import AllometricFit, build_dsl_map, fit_power_law
from .config import TOMConfig
from .maps import build_extent_map, build_maps, coverage_summary
from .pipeline import register_cohort
from .template import TemplateSpace, build_template, load_template

try:  # pragma: no cover - only meaningful for installed package
    __version__ = version("psma-tom")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "TOMConfig",
    "AllometricFit",
    "fit_power_law",
    "build_dsl_map",
    "TemplateSpace",
    "build_template",
    "load_template",
    "register_cohort",
    "build_maps",
    "build_extent_map",
    "coverage_summary",
    "__version__",
]
