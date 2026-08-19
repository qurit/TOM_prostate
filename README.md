# psma-tom

Population-based multi-parametric **Tumor Occurrence Maps (TOM)** of metastatic
distribution in prostate cancer from PSMA PET/CT.

Given a cohort of PSMA PET/CT scans with lesion segmentations, this package maps
every patient into a common template space and accumulates their disease into
population-level maps that answer four different questions about the same voxel:

| Map | Quantity | Question |
|---|---|---|
| `TOM_occurrence` | P(lesion at *v*) | Where does metastatic disease concentrate? |
| `TOM_suv_mean` | E[SUV \| lesion at *v*] | How avid is disease when it occurs here? |
| `TOM_extent_voxels` | E[component extent \| lesion at *v*] | How extensive is it when it occurs here? |
| `TOM_dsl` | E[deviation from TLA = a·MTV^b \| lesion at *v*] | Is disease here more avid than its bulk predicts? |

Two supporting maps are written alongside: `TOM_suv_cv` (voxel-wise coefficient
of variation of lesion SUV) and `TOM_lesion_count` (how many patients
contributed to each voxel — the denominator behind every conditional map).

All maps share the template CT grid, so they can be overlaid, thresholded
together, and used as independent spatial priors.

---

## Install

```bash
git clone https://github.com/USER/psma-tom.git
cd psma-tom
pip install -e .                        # core: CPU, affine registration
pip install -e ".[deformable]"          # + uniGradICON (needs torch, itk)
pip install -e ".[interactive,dev]"     # + plotly 3D views, test tooling
```

Python ≥ 3.10. The deformable backend downloads ~200 MB of model weights on
first use and is roughly 8× faster on a GPU, but the pipeline runs end to end on
CPU with `registration: affine`.

## Quick start

Prepare a CSV with one row per patient (see `data/cases_psma.example.csv`):

```csv
case_id,ct_path,mask_path,pet_path
psma_0001,/data/psma_0001_0000.nii.gz,/data/labels/psma_0001.nii.gz,/data/psma_0001_0001.nii.gz
```

Then:

```bash
psma-tom -c configs/default.yaml check       # validate paths before anything expensive
psma-tom -c configs/default.yaml template    # pick and write the template space
psma-tom -c configs/default.yaml register    # register the cohort (resumable)
psma-tom -c configs/default.yaml maps        # accumulate the population maps
psma-tom -c configs/default.yaml allometry   # fit TLA = a·MTV^b, build the DSL map
psma-tom -c configs/default.yaml figures --interactive
```

Or from Python:

```python
from psma_tom import TOMConfig, build_template, register_cohort, build_maps

config = TOMConfig.from_yaml("configs/default.yaml")
template = build_template(config)
report = register_cohort(config, template=template)   # resumable
maps = build_maps(config, template=template)
print(report.summary(), maps.describe())
```

`notebooks/01_build_multiparametric_tom.ipynb` runs the same sequence on Colab
with Drive mounted. It contains no pipeline logic — only configuration and
calls into the package — so any figure it produces traces back to a versioned
function.

## How it works

**1 · Template selection.** The case whose voxel count is closest to the cohort
median becomes the template. Only NIfTI headers are read, so scanning several
hundred whole-body CTs takes seconds instead of tens of minutes.

**2 · Lesion-level scalar maps.** For each patient, connected-component
labelling turns the binary mask into maps where every voxel of a lesion carries
that lesion's mean SUV, its volume, and its extent. These are computed at the
original acquisition resolution — downsampling first would measure the
downsampled blobs rather than the lesions.

**3 · Registration.** Three stages: body-centroid translation, then a
uniGradICON deformable warp, with an affine + Mattes MI fallback when the
centroid shift leaves no overlap between the fields of view. The mask, SUV and
volume maps are warped with the *same* transform as the CT, so lesion-level
statistics stay attached to their anatomy. Which path each case took is recorded
in `registration_report.json`.

**4 · Accumulation.** A single streaming pass over the registered files computes
sums, sums of squares and component extents together. Peak memory is a handful
of template-sized arrays regardless of cohort size.

### Resumability

Every case writes its outputs the moment they exist, and the loop skips any case
whose three files are already on disk. A run that dies at case 300 of 472 —
hosted-runtime disconnect, OOM, full disk — resumes by being started again.
There is no bookkeeping file to fall out of sync with reality; the outputs
directory *is* the state.

### Why extent, not volume

At an 8 mm template grid one voxel is ≈ 2 cm³, while the median PSMA lesion is
smaller than that. A "conditional mean volume" at this resolution mostly
measures the voxel size, not the disease. Component extent in voxels is the
honest quantity, and the figure produced by
`viz.plot_lesion_size_distribution` is the evidence for that choice.
`TOM_volume_mean` is still written, and is still documented as not
interpretable at this grid.

### Allometric deviation (DSL)

Total lesion activity scales with metabolic tumour volume as a power law,
`TLA = a · MTV^b`, fitted by least squares in log-log space. A patient's signed
log₁₀ residual from that law — more or less avid than their bulk predicts — is
averaged over the voxels where they have disease to give `TOM_dsl`. The fit is
always estimated from the cohort passed in; constants from a previous cohort are
never assumed, though a fitted `AllometricFit` can be applied to another cohort
explicitly.

## Configuration

Every parameter lives in a YAML file, so a run reproduces from one artifact.

```yaml
case_csv:   data/cases_psma.csv
output_dir: results_psma_TOM_multi

shrink_factor:     4            # 2 mm source data -> 8 mm TOM grid
min_lesion_voxels: 3            # noise filter at full resolution

registration:               unigradicon   # or: affine
unigradicon_finetune_steps: 0             # 0 = inference only

occurrence_threshold: 0.01
```

Three presets ship in `configs/`: `default.yaml`, `colab.yaml` (Drive paths),
and `affine.yaml` (CPU-only, no model download — useful for a fast smoke test).

Outputs are tagged by backend (`ug_masks/` vs `affine_masks/`), so the two can
never silently overwrite each other.

## Outputs

```
results_psma_TOM_multi/
├── TOM_template_ct.nii.gz          full-resolution template CT
├── TOM_template_ct_ds.nii.gz       the TOM grid — send this with any map
├── TOM_body_mask.nii.gz
├── ug_masks/  ug_suv_maps/  ug_vol_maps/     per-case checkpoints
├── TOM_occurrence.nii.gz
├── TOM_suv_mean.nii.gz
├── TOM_suv_cv.nii.gz
├── TOM_extent_voxels.nii.gz
├── TOM_dsl.nii.gz
├── TOM_lesion_count.nii.gz
├── TOM_coverage_summary.csv
├── cohort_burden_metrics.csv
├── registration_report.json
└── figures/
```

A TOM NIfTI is meaningless without the CT that defines its space. **Always share
`TOM_template_ct_ds.nii.gz` alongside any map** — without it the recipient has
nothing to overlay it on.

## Development

```bash
pip install -e ".[dev]"
pytest                 # ~40 tests, seconds, no real data required
ruff check src tests
black --check src tests
```

Patient data is never redistributable, so the test suite builds synthetic
PET/CT phantoms and exercises the real code paths against them: that occurrence
is a probability, that conditional means are conditional on lesion presence and
not diluted by the whole cohort, that the CV matches its definition, that extent
counts voxels rather than physical volume, that registration improves alignment
on a known shift, and that a second pipeline invocation is a no-op.

`.gitignore` excludes `*.nii.gz` and `data/*.csv` so that images and case tables
with patient paths cannot be committed by accident.

## Citation

If you use this code, please cite the accompanying manuscript 
F. Yousefirizi, P. Esquinas, C. Uribe, Population-Based Multi-Parametric Tumor Occurrence Maps of Metastatic Distribution in Prostate Cancer Using PSMA PET/CT, submitted to JNM

## License

MIT — see `LICENSE`.
