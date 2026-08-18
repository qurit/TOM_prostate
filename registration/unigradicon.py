"""Three-stage deformable registration built on uniGradICON.

    stage 1  body-centroid translation      (coarse, removes FOV offset)
    stage 2  uniGradICON deformable warp    (foundation model, no per-case fit)
    fallback affine + Mattes MI             (when stage 1 leaves no overlap)

The optional dependencies (``itk``, ``torch``, ``unigradicon``) are imported
lazily so that the rest of the package — and the whole test suite — runs
without a GPU or a 200 MB model download.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import SimpleITK as sitk

from ..preprocessing import body_centroid, to_unit_range
from .affine import AffineRegistrar
from .base import Alignment, Registrar

log = logging.getLogger(__name__)


# ── SimpleITK <-> ITK bridges ────────────────────────────────────────────────
def sitk_to_itk(img: sitk.Image):
    """Convert to ``itk.Image[F, 3]``, preserving spacing, origin, direction."""
    import itk

    arr = sitk.GetArrayFromImage(sitk.Cast(img, sitk.sitkFloat32)).astype(np.float32)
    out = itk.GetImageFromArray(arr)
    out.SetSpacing([float(s) for s in img.GetSpacing()])
    out.SetOrigin([float(o) for o in img.GetOrigin()])
    direction = np.asarray(img.GetDirection()).reshape(3, 3)
    # ITK stores the direction cosines transposed relative to SimpleITK.
    out.SetDirection(itk.matrix_from_array(direction.T))
    return itk.CastImageFilter[type(out), itk.Image.F3].New()(out)


def itk_to_sitk(img) -> sitk.Image:
    import itk

    arr = itk.GetArrayFromImage(img).astype(np.float32)
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing([float(s) for s in img.GetSpacing()])
    out.SetOrigin([float(o) for o in img.GetOrigin()])
    return out


class UniGradICONRegistrar(Registrar):
    name = "unigradicon"

    def __init__(
        self,
        shrink: int = 2,
        finetune_steps: int = 0,
        hu_threshold: int = -300,
        fallback: Registrar | None = None,
    ) -> None:
        self.shrink = shrink
        self.finetune_steps = finetune_steps
        self.hu_threshold = hu_threshold
        self.fallback = fallback or AffineRegistrar()
        self._model = None

    # ── model ────────────────────────────────────────────────────────────────
    @property
    def model(self):
        """Load the pretrained network once per process."""
        if self._model is None:
            import unigradicon

            log.info("loading uniGradICON weights (first call downloads ~200 MB)")
            model = unigradicon.get_unigradicon()
            model.eval()
            self._model = model
        return self._model

    # ── registration ─────────────────────────────────────────────────────────
    def fit(self, fixed: sitk.Image, moving: sitk.Image) -> Alignment:
        import icon_registration.itk_wrapper as itk_wrapper
        import itk
        import torch

        translation = self._centroid_translation(fixed, moving)
        prealigned = sitk.Resample(
            sitk.Cast(moving, sitk.sitkFloat32),
            fixed,
            translation,
            sitk.sitkLinear,
            -1024.0,
            sitk.sitkFloat32,
        )

        fixed_ds = sitk.Shrink(fixed, [self.shrink] * 3)
        moving_ds = sitk.Shrink(prealigned, [self.shrink] * 3)

        fixed_itk = sitk_to_itk(to_unit_range(fixed_ds))
        moving_itk = sitk_to_itk(to_unit_range(moving_ds))

        # A constant moving image means the centroid shift pushed the patient
        # entirely outside the template FOV. register_pair would return a
        # meaningless field, so take the affine path instead.
        moving_arr = itk.GetArrayFromImage(moving_itk)
        if float(moving_arr.max()) == float(moving_arr.min()):
            del fixed_itk, moving_itk, fixed_ds, moving_ds, prealigned
            gc.collect()
            log.info("no overlap after centroid translation — affine fallback")
            alignment = self.fallback.fit(fixed, moving)
            alignment.method = "affine_fallback"
            return alignment

        with torch.no_grad():
            phi_AB, phi_BA = itk_wrapper.register_pair(
                self.model,
                fixed_itk,
                moving_itk,
                finetune_steps=self.finetune_steps if self.finetune_steps > 0 else None,
            )

        del fixed_itk, moving_itk, fixed_ds, moving_ds, prealigned
        self._release()

        return Alignment(
            method="unigradicon",
            transform=phi_BA,
            metadata={"phi_AB": phi_AB, "translation": list(translation.GetOffset())},
        )

    def apply(
        self,
        alignment: Alignment,
        image: sitk.Image,
        reference: sitk.Image,
        interp: str = "nearest",
        default_value: float = 0.0,
        out_type: int | None = None,
    ) -> sitk.Image:
        if alignment.method != "unigradicon":
            return self.fallback.apply(alignment, image, reference, interp, default_value, out_type)

        import itk

        src = sitk_to_itk(image)
        ref = sitk_to_itk(reference)
        image_type = type(src)

        interpolator = (
            itk.LinearInterpolateImageFunction[image_type, itk.D].New()
            if interp == "linear"
            else itk.NearestNeighborInterpolateImageFunction[image_type, itk.D].New()
        )

        resampler = itk.ResampleImageFilter[image_type, image_type].New()
        resampler.SetInput(src)
        resampler.SetTransform(alignment.transform)
        resampler.SetReferenceImage(ref)
        resampler.UseReferenceImageOn()
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(float(default_value))
        resampler.Update()

        warped = itk_to_sitk(resampler.GetOutput())
        # The ITK round-trip drops the direction matrix; restore the template's.
        warped.CopyInformation(reference)
        return sitk.Cast(warped, out_type) if out_type is not None else warped

    # ── internals ────────────────────────────────────────────────────────────
    def _centroid_translation(
        self, fixed: sitk.Image, moving: sitk.Image
    ) -> sitk.TranslationTransform:
        offset = body_centroid(fixed, self.hu_threshold) - body_centroid(moving, self.hu_threshold)
        return sitk.TranslationTransform(3, offset.tolist())

    @staticmethod
    def _release() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
