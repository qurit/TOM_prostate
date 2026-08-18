"""Command-line entry point.

psma-tom check      validate the case table and referenced files
psma-tom template   select and write the template space
psma-tom register   register the cohort (resumable)
psma-tom maps       accumulate registered cases into TOM maps
psma-tom allometry  fit TLA = a * MTV^b and build the DSL map
psma-tom figures    render the paper figures
psma-tom all        template -> register -> maps -> figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import TOMConfig

log = logging.getLogger("psma_tom")


def _load_config(args: argparse.Namespace) -> TOMConfig:
    config = TOMConfig.from_yaml(args.config) if args.config else TOMConfig()
    if args.case_csv:
        config.case_csv = Path(args.case_csv)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.registration:
        config.registration = args.registration
    config.__post_init__()
    return config.ensure_dirs()


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    from .io import check_cases_exist, load_cases

    config = _load_config(args)
    df = load_cases(config)
    print(f"cases       : {len(df)}")
    print(f"columns     : {list(df.columns)}")

    bad = check_cases_exist(config, df, limit=args.limit)
    if bad:
        print(f"missing files for {len(bad)} case(s): {bad[:10]}")
        return 1
    print("all referenced CT / mask / PET files exist")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    from .template import build_template

    config = _load_config(args)
    template = build_template(config, case_id=args.case_id)
    print(template.describe())
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    from .pipeline import register_cohort

    config = _load_config(args)
    report = register_cohort(config, limit=args.limit)
    print(report.summary())
    return 1 if report.failed else 0


def cmd_maps(args: argparse.Namespace) -> int:
    from .maps import build_maps, coverage_summary
    from .template import load_template

    config = _load_config(args)
    template = load_template(config)
    maps = build_maps(config, template=template)
    print(maps.describe())
    print(coverage_summary(config, maps=maps, template=template).to_string(index=False))
    return 0


def cmd_allometry(args: argparse.Namespace) -> int:
    from .allometry import build_dsl_map
    from .template import load_template

    config = _load_config(args)
    fit, _ = build_dsl_map(config, template=load_template(config))
    print(fit.describe())
    print(f"DSL map written to {config.output_dir / 'TOM_dsl.nii.gz'}")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    import SimpleITK as sitk

    from . import viz
    from .maps import build_maps
    from .template import load_template

    config = _load_config(args)
    template = load_template(config)
    maps = build_maps(config, template=template, save=False)
    spacing = template.spacing

    viz.plot_tom_panels(
        maps,
        spacing,
        output=config.figure_dir / "TOM_three_panel.png",
        occurrence_threshold=config.occurrence_threshold,
        suv_cap=config.suv_display_cap,
    )
    viz.plot_three_views(
        maps,
        spacing,
        output=config.figure_dir / "TOM_three_views.png",
        occurrence_threshold=config.occurrence_threshold,
        suv_cap=config.suv_display_cap,
    )
    if args.interactive:
        ct = sitk.GetArrayFromImage(sitk.Cast(template.downsampled, sitk.sitkFloat32))
        viz.interactive_3d(maps, spacing, config.figure_dir / "TOM_3D.html", template_ct=ct)
    print(f"figures written to {config.figure_dir}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_template, cmd_register, cmd_maps, cmd_figures):
        code = step(args)
        if code and step is not cmd_register:
            return code
    return 0


# ── parser ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psma-tom", description=__doc__)
    parser.add_argument("-c", "--config", help="YAML config file")
    parser.add_argument("--case-csv", help="override case table path")
    parser.add_argument("--output-dir", help="override output directory")
    parser.add_argument(
        "--registration", choices=("unigradicon", "affine"), help="override backend"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="validate the case table")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("template", help="build the template space")
    p.add_argument("--case-id", default=None, help="use this case instead of the median")
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("register", help="register the cohort (resumable)")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("maps", help="accumulate TOM maps")
    p.set_defaults(func=cmd_maps)

    p = sub.add_parser("allometry", help="fit the scaling law and build the DSL map")
    p.set_defaults(func=cmd_allometry)

    p = sub.add_parser("figures", help="render figures")
    p.add_argument("--interactive", action="store_true", help="also write the 3D HTML view")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("all", help="run the whole pipeline")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--case-id", default=None)
    p.add_argument("--interactive", action="store_true")
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
