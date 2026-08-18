"""Figures: maximum-intensity projections, three-panel summaries, 3D views.

Projections are drawn with the physically correct aspect ratio (derived from
the template spacing) — a TOM grid is anisotropic, and plotting it with a
default aspect silently stretches the anatomy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

log = logging.getLogger(__name__)

VIEWS = {"coronal": 1, "sagittal": 2, "axial": 0}
EXTENT_COLORS = ["black", "#4CAF50", "#FF9800", "#C62828"]


def mip(array: np.ndarray, view: str = "coronal") -> np.ndarray:
    """Maximum-intensity projection with the head at the top."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}, got {view!r}")
    projection = array.max(axis=VIEWS[view])
    return projection if view == "axial" else np.flipud(projection)


def aspect_for(spacing: tuple[float, ...], view: str = "coronal") -> float:
    """imshow aspect that makes one display pixel physically square."""
    sx, sy, sz = spacing
    return {"coronal": sz / sx, "sagittal": sz / sy, "axial": sy / sx}[view]


def _fmt_volume(cm3: float) -> str:
    """Format a volume with enough precision to stay meaningful.

    A fixed "%.0f" turns any sub-millilitre threshold into "0 cm3", which
    reads as nonsense on a legend ("Focal <0 cm3").
    """
    if cm3 >= 10:
        return f"{cm3:.0f}"
    if cm3 >= 1:
        return f"{cm3:.1f}"
    return f"{cm3:.2f}"


def extent_categories(extent_mip: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Tercile-split the extent map into focal / moderate / extensive."""
    values = extent_mip[mask & (extent_mip > 0)]
    if values.size == 0:
        return np.zeros_like(extent_mip), 0.0, 0.0

    p33, p66 = np.percentile(values, [33, 66])
    categories = np.zeros_like(extent_mip, dtype=np.float32)
    categories[(extent_mip > 0) & (extent_mip <= p33)] = 1
    categories[(extent_mip > p33) & (extent_mip <= p66)] = 2
    categories[extent_mip > p66] = 3
    categories[~mask] = 0
    return categories, float(p33), float(p66)


def plot_tom_panels(
    maps,
    spacing: tuple[float, ...],
    output: Path | None = None,
    view: str = "coronal",
    occurrence_threshold: float = 0.01,
    suv_cap: float = 25.0,
    dpi: int = 300,
):
    """Main paper figure: occurrence, conditional mean SUV, disease extent."""
    aspect = aspect_for(spacing, view)
    voxel_cm3 = float(np.prod(spacing)) / 1000.0

    occurrence = mip(maps.occurrence, view)
    suv = np.clip(mip(maps.suv_mean, view), 0, suv_cap)
    extent = mip(maps.extent, view)

    keep = occurrence > occurrence_threshold
    suv[~keep] = 0
    extent[~keep] = 0
    categories, p33, p66 = extent_categories(extent, keep)

    fig, axes = plt.subplots(1, 3, figsize=(16, 12))

    im0 = axes[0].imshow(
        occurrence, cmap="hot", vmin=0, vmax=occurrence.max(), origin="upper", aspect=aspect
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.04, label="P(lesion)")
    axes[0].set_title("Occurrence probability\nP(lesion | voxel)", fontsize=11)

    im1 = axes[1].imshow(
        np.ma.masked_where(suv < 0.1, suv),
        cmap="inferno",
        vmin=0,
        vmax=suv_cap,
        origin="upper",
        aspect=aspect,
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.04, label="Mean SUV")
    axes[1].set_title("Conditional mean SUV\nE[SUV | lesion at voxel]", fontsize=11)

    cmap = ListedColormap(EXTENT_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im2 = axes[2].imshow(categories, cmap=cmap, norm=norm, origin="upper", aspect=aspect)
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.04, ticks=[0, 1, 2, 3])
    low, high = _fmt_volume(p33 * voxel_cm3), _fmt_volume(p66 * voxel_cm3)
    cbar.set_ticklabels(
        [
            "none",
            f"Focal\n≤{low} cm³",
            f"Moderate\n{low}–{high} cm³",
            f"Extensive\n>{high} cm³",
        ]
    )
    cbar.ax.tick_params(labelsize=9)
    axes[2].set_title("Disease extent\nE[spatial extent | lesion at voxel]", fontsize=11)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Multi-parametric PSMA tumor occurrence maps — "
        f"{view} MIP\nN={maps.n_cases} patients  ·  P(lesion) > {occurrence_threshold}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, output, dpi)
    return fig


def plot_three_views(
    maps,
    spacing: tuple[float, ...],
    output: Path | None = None,
    occurrence_threshold: float = 0.01,
    suv_cap: float = 25.0,
    dpi: int = 300,
):
    """Occurrence (top row) and mean SUV (bottom) in all three orientations."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 14))

    for col, view in enumerate(("coronal", "sagittal", "axial")):
        aspect = aspect_for(spacing, view)
        occurrence = mip(maps.occurrence, view)
        suv = mip(maps.suv_mean, view)
        keep = occurrence > occurrence_threshold

        im0 = axes[0, col].imshow(
            occurrence, cmap="hot", vmin=0, vmax=occurrence.max(), origin="upper", aspect=aspect
        )
        plt.colorbar(im0, ax=axes[0, col], fraction=0.04, label="P(lesion)")
        axes[0, col].set_title(f"Occurrence — {view} MIP", fontsize=10)

        im1 = axes[1, col].imshow(
            np.ma.masked_where(~keep, np.clip(suv, 0, suv_cap)),
            cmap="inferno",
            vmin=0,
            vmax=suv_cap,
            origin="upper",
            aspect=aspect,
        )
        plt.colorbar(im1, ax=axes[1, col], fraction=0.04, label="Mean SUV")
        axes[1, col].set_title(f"Mean SUV — {view} MIP", fontsize=10)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"PSMA tumor occurrence maps (N={maps.n_cases})\n"
        "Top: P(lesion | voxel)  ·  Bottom: E[SUV | lesion at voxel]",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, output, dpi)
    return fig


def plot_registration_qc(
    template: np.ndarray,
    warped: dict,
    output: Path | None = None,
    dpi: int = 150,
):
    """Side-by-side mid-axial slices: template vs each registration method."""
    panels = [("Template CT", template)] + list(warped.items())
    z = template.shape[0] // 2

    def window(arr):
        sl = arr[z].astype(np.float32)
        lo, hi = np.percentile(sl, [2, 98])
        return np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for ax, (label, arr) in zip(axes, panels, strict=True):
        ax.imshow(window(arr), cmap="gray", origin="lower")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Registration QC (mid-axial slice z={z})", fontsize=11)
    fig.tight_layout()
    _save(fig, output, dpi)
    return fig


def plot_lesion_size_distribution(
    sizes_cm3: np.ndarray,
    voxel_volume_cm3: float,
    output: Path | None = None,
    dpi: int = 200,
):
    """Lesion sizes against the TOM voxel size — the resolution-limit figure.

    This is the plot that justifies reporting extent in voxels rather than a
    conditional mean volume.
    """
    sizes = np.asarray(sizes_cm3, dtype=float)
    percentiles = np.percentile(sizes, [10, 25, 50, 75, 90])
    labels = [f"P{p}\n{v:.1f} cm³" for p, v in zip((10, 25, 50, 75, 90), percentiles, strict=True)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(sizes[sizes < 50], bins=50, color="#028090", edgecolor="white")
    axes[0].axvline(
        float(np.median(sizes)),
        color="#C94B3B",
        linestyle="--",
        label=f"median = {np.median(sizes):.1f} cm³",
    )
    axes[0].axvline(
        voxel_volume_cm3,
        color="navy",
        linestyle="--",
        label=f"1 TOM voxel = {voxel_volume_cm3:.2f} cm³",
    )
    axes[0].set_xlabel("Lesion volume (cm³)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Lesion volumes < 50 cm³  (n = {sizes.size})")
    axes[0].legend(fontsize=9)

    below = 100.0 * float((sizes < voxel_volume_cm3).mean())
    bars = axes[1].bar(
        labels,
        percentiles,
        color=["#4CAF50", "#4CAF50", "#FF9800", "#FF9800", "#C62828"],
        edgecolor="white",
        width=0.6,
    )
    axes[1].axhline(
        voxel_volume_cm3,
        color="navy",
        linestyle="--",
        label=f"1 TOM voxel = {voxel_volume_cm3:.2f} cm³",
    )
    axes[1].set_ylabel("Lesion volume (cm³)")
    axes[1].set_title(f"{below:.0f}% of lesions are smaller than one TOM voxel")
    axes[1].legend(fontsize=9)
    for bar, value in zip(bars, percentiles, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle("PSMA lesion size distribution vs TOM resolution", fontsize=12)
    fig.tight_layout()
    _save(fig, output, dpi)
    return fig


def interactive_3d(
    maps,
    spacing: tuple[float, ...],
    output: Path,
    occurrence_threshold: float = 0.015,
    suv_cap: float = 25.0,
    template_ct: np.ndarray | None = None,
):
    """Write a standalone interactive 3D HTML view.

    Colour encodes conditional mean SUV, marker size encodes disease extent, so
    all three maps are readable in one scene.
    """
    import plotly.graph_objects as go

    sx, sy, sz = spacing
    selected = maps.occurrence > occurrence_threshold
    if not selected.any():
        raise ValueError(f"no voxels above occurrence threshold {occurrence_threshold}")

    zz, yy, xx = np.where(selected)
    occurrence = maps.occurrence[selected]
    suv = np.clip(maps.suv_mean, 0, suv_cap)[selected]
    extent = maps.extent[selected]

    lo, hi = max(float(extent.min()), 1.0), max(float(extent.max()), 2.0)
    size = np.clip(3 + 13 * (extent - lo) / (hi - lo + 1e-8), 3, 16)
    p33, p66 = np.percentile(extent, [33, 66])

    fig = go.Figure()

    if template_ct is not None:
        ct = (template_ct - template_ct.min()) / (np.ptp(template_ct) + 1e-8)
        Z, Y, X = ct.shape
        level = float(np.percentile(ct[ct > 0.01], 20))
        fig.add_trace(
            go.Isosurface(
                x=np.tile(np.arange(X) * sx, Z * Y),
                y=np.tile(np.repeat(np.arange(Y) * sy, X), Z),
                z=np.repeat(np.arange(Z) * sz, Y * X),
                value=ct.ravel(),
                isomin=level - 0.01,
                isomax=level + 0.01,
                surface_count=1,
                colorscale=[[0, "rgb(180,200,220)"], [1, "rgb(200,215,230)"]],
                showscale=False,
                opacity=0.07,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Body outline",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=xx * sx,
            y=yy * sy,
            z=zz * sz,
            mode="markers",
            marker=dict(
                size=size,
                color=suv,
                colorscale="inferno",
                cmin=0,
                cmax=suv_cap,
                opacity=0.9,
                colorbar=dict(title=dict(text="Mean SUV", side="right"), x=1.02, len=0.6),
                line=dict(width=0),
            ),
            customdata=np.stack([occurrence, suv, extent], axis=1),
            hovertemplate=(
                "<b>P(lesion)</b>: %{customdata[0]:.3f}<br>"
                "<b>Mean SUV</b>: %{customdata[1]:.1f}<br>"
                "<b>Extent</b>: %{customdata[2]:.0f} voxels<extra></extra>"
            ),
            name="TOM",
            showlegend=False,
        )
    )

    for label, marker_size in (
        (f"Focal (≤{p33:.0f} vox)", 4),
        (f"Moderate ({p33:.0f}–{p66:.0f} vox)", 9),
        (f"Extensive (>{p66:.0f} vox)", 15),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=marker_size, color="white", opacity=0.9),
                name=label,
            )
        )

    axis = dict(showgrid=False, zeroline=False, color="white")
    fig.update_layout(
        title=dict(
            text=(
                "PSMA multi-parametric TOM — 3D<br><sup>"
                f"colour = mean SUV · size = disease extent · P(lesion) > {occurrence_threshold} · "
                f"N={maps.n_cases}</sup>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="x (mm)", **axis),
            yaxis=dict(title="y (mm)", **axis),
            zaxis=dict(title="z (mm)", **axis),
            bgcolor="black",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.5)),
        ),
        paper_bgcolor="black",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.7)", bordercolor="grey", borderwidth=1, x=0.0, y=0.5),
        width=1000,
        height=750,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    log.info("wrote %s", output)
    return output


def _save(fig, output: Path | None, dpi: int) -> None:
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    log.info("wrote %s", output)
