#!/usr/bin/env python
from __future__ import annotations

from typing import TYPE_CHECKING

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.coaddBase as coaddBase
import lsst.utils as utils
import numpy as np
from descwl_coadd import DEFAULT_INTERP, FLAGS2INTERP, warp_exposures
from lsst.daf.butler import DeferredDatasetHandle
from lsst.meas.algorithms import CoaddPsf
from lsst.pex.config import (
    ConfigField,
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
    RangeField,
)
from lsst.pipe.base import (
    Instrument,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddInputRecorder import CoaddInputRecorderTask
from lsst.skymap import BaseSkyMap

if TYPE_CHECKING:
    from lsst.afw.exposure import Exposure, ExposureCatalog


class MakeShearWarpConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    defaultTemplates={"calexpType": ""},
):
    calexps = Input(
        doc="Input exposures to be resampled onto a SkyMap projection/patch",
        name="{calexpType}calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )

    skyMap = Input(
        doc="Input definition of geometry/bbox and projection/wcs for warped "
        "exposures."
        "This must be cell-based.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    visit_summary = Input(
        doc="Input visit-summary catalog with updated calibration objects.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=(
            "instrument",
            "visit",
        ),
    )

    warp = Output(
        doc="Output direct warped exposure produced by resampling "
        "calexps onto the skyMap patch geometry.",
        name="{calexpType}calexp_descwarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )

    mfrac_warp = Output(
        doc="Output mask fraction warped exposure",
        name="{calexpType}calexp_mfrac_descwarp",
        storageClass="ImageF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)  # Harmless and a no-op for now.
        if not config:
            return

        # Remove the visit_summary connection if it is not going to be used.
        if not config.use_visit_summary:
            del self.visit_summary

        # Dynamically set output connections for noise images depending on the
        # number of noise realizations specified in the config.
        for n in range(config.num_noise_realizations):
            noise_warp = Output(
                doc="Output direct warped noise image produced by resampling"
                "resampling calexps onto the skyMap patch geometry.",
                name=f"{config.connections.calexpType}calexp_noise{n}_descwarp",
                # Store it as a MaskedImage to preserve the variance plane.
                storageClass="MaskedImageF",
                dimensions=("tract", "patch", "skymap", "visit", "instrument"),
            )
            setattr(self, f"noise{n}_warp", noise_warp)


class MakeShearWarpConfig(
    PipelineTaskConfig,
    pipelineConnections=MakeShearWarpConnections,
):
    """Configuration for the MakeShearWarpTask."""

    MAX_NUM_NOISE_REALIZATIONS = 3
    """
    num_noise_realizations is defined as a RangeField to prevent from making
    multiple output connections and blowing up the memory usage by accident.
    An upper bound of 3 is based on the best guess of the maximum number of
    noise realizations that will be used for metadetection.
    """

    num_noise_realizations = RangeField[int](
        default=1,
        doc=(
            "Number of noise realizations to simulate and persist."
            "Currently, the only values accepted are 0 and 1."
        ),
        min=0,
        max=MAX_NUM_NOISE_REALIZATIONS,
        inclusiveMax=True,
    )

    seed_offset = Field[int](
        default=0,
        doc="Offset to add to the seed used to generate noise realizations",
    )

    remove_poisson = Field[bool](
        doc="Should the Poisson noise contribution be removed from the "
        "variance estimate?",
        default=False,
    )

    warper = ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Configuration for the warper that warps the image and noise",
    )

    mfrac_warper = ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Configuration for the warp that warps the mask fraction image",
    )

    bad_mask_planes = ListField[str](
        doc="Mask planes that count towards the masked fraction within a cell.",
        default=("BAD", "CR", "SAT"),
    )

    use_visit_summary = Field[bool](
        doc="Whether to use the visit summary table to update the calibration "
        "(PSF, WCS, photometric calibration, aperture correction maps) "
        "of the input exposures? This should be True for best results but "
        "can be set to False for testing purposes.",
        default=True,
    )

    inputRecorder = ConfigurableField(
        doc="Subtask that helps fill CoaddInputs catalogs added to the final Exposure",
        target=CoaddInputRecorderTask,
    )

    def setDefaults(self) -> None:
        super().setDefaults()

        # Set DESC shear specific values here.
        self.bad_mask_planes = FLAGS2INTERP
        # Configure the warpers to have the same values as in descwl_coadd.
        self.warper.warpingKernelName = DEFAULT_INTERP
        self.mfrac_warper.warpingKernelName = "bilinear"

    def validate(self) -> None:
        # For now, we limit this value to be at most 1 because the DESC code
        # does not return multiple noise realizations without also warping
        # the image multiple times. The task is also not (yet) equipped to
        # handle multiple noise realizations.
        if self.num_noise_realizations > 1:
            raise FieldValidationError("num_noise_realizations > 1 not yet supported")

        super().validate()


class MakeShearWarpTask(PipelineTask):
    """
    This Task differs from the standard `MakeWarp` Task in the following
    ways:

    1. No selection on ccds at the time of warping. This is done later during
       the coaddition stage.
    2. Interpolate over a set of masked pixels before warping.
    3. Generate an image where each pixel denotes how much of the pixel is
       masked.
    4. Generate a noise warp with the same interpolation applied.
    """

    ConfigClass = MakeShearWarpConfig
    _DefaultName = "descwarp"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warper)
        self.mfrac_warper = afwMath.Warper.fromConfig(self.config.mfrac_warper)
        self.makeSubtask("inputRecorder")

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        # Docstring inherited.

        # Read in all inputs.
        inputs = butlerQC.get(inputRefs)

        if not inputs["calexps"]:
            raise pipeBase.NoWorkFound("No input warps provided for co-addition")

        # Construct skyInfo expected by `run`.  We remove the SkyMap itself
        # from the dictionary so we can pass it as kwargs later.
        skyMap = inputs.pop("skyMap")
        if not skyMap.config.tractBuilder.name == "cells":
            self.log.warning(
                "SkyMap is not cell-based; "
                "The output warps cannot be used to construct cell coadds."
            )

        quantumDataId = butlerQC.quantum.dataId
        skyInfo = coaddBase.makeSkyInfo(
            skyMap, tractId=quantumDataId["tract"], patchId=quantumDataId["patch"]
        )

        visit_summary = (
            inputs["visit_summary"] if self.config.use_visit_summary else None
        )

        results = self.run(
            exposures=inputs["calexps"],
            visit_summary=visit_summary,
            skyInfo=skyInfo,
        )
        butlerQC.put(results, outputRefs)

    def run(
        self,
        exposures: list[DeferredDatasetHandle],
        skyInfo: pipeBase.Struct,
        visit_summary: ExposureCatalog | None,
    ) -> pipeBase.Struct:
        """Run the task on a set of DeferredDatasetHandle of exposures.

        Parameters
        ----------
        exposures : `list` [`~lsst.daf.butler.DeferredDatasetHandle`]
            List of exposures (calexp s) to be warped.
        skyInfo : `~pipeBase.Struct`
            A Struct object containing wcs, bounding box, and other information
            about the patches within the tract.
        visit_summary : `~lsst.afw.table.ExposureCatalog` | None
            Table of visit summary information.  If provided, the visit summary
            information will be used to update the calibration of the input
            exposures.  If None, the input exposures will be used as-is.

        Returns
        -------
        results : `~pipeBase.Struct`
            A Struct object containing the warped exposure, noise exposure(s),
            and masked fraction image.

        Notes
        -----
        We do not support ``exposures`` to be a list of
        `~lsst.afw.image.Exposure` because the metadata that will be required
        to generate the seeds require the data coordinates defined in the
        middleware, that are not available from the exposures themselves.
        """
        coadd_bbox, coadd_wcs = skyInfo.bbox, skyInfo.wcs

        # Initialze the objects that will hold the warped images.
        final_warp = afwImage.ExposureF(coadd_bbox, coadd_wcs)
        # Set the image to nan, variance to infinity and mask to NO_DATA.
        final_warp.getMaskedImage().set(
            np.nan,
            afwImage.Mask.getPlaneBitMask("NO_DATA"),
            np.inf,
        )
        final_noise_warp = afwImage.ExposureF(coadd_bbox, coadd_wcs)
        final_mfrac_warp = afwImage.ExposureF(coadd_bbox, coadd_wcs)

        visit_id = exposures[0].dataId["visit"]

        inputRecorder = self.inputRecorder.makeCoaddTempExpRecorder(
            visit_id,
            len(exposures),
        )

        total_good_pixels = 0

        for iexp, expRef in enumerate(exposures):
            self.log.info(
                "Warping exposure %d/%d for id=%s",
                iexp + 1,
                len(exposures),
                visit_id,
            )
            # TODO: We might be able to speed up warping (not I/O) by getting
            # only the part of exposure that would overlaps with the coadd
            # bbox, with some buffer around it, specified by the interpLength.
            exp = expRef.get()

            # Update the exposure with the latest and greatest calibration.
            self._apply_all_calibrations(exp, visit_summary)

            data_id = expRef.dataId
            self.log.debug("Exposure data id = %s", data_id)

            seed = self.get_seed_from_data_id(data_id)
            self.log.debug(
                "Using seed %d to generate noise for %s",
                seed + self.config.seed_offset,
                data_id,
            )
            rng = np.random.RandomState(seed + self.config.seed_offset)

            warped_exposures = warp_exposures(
                exp,
                coadd_wcs,
                coadd_bbox,
                remove_poisson=self.config.remove_poisson,
                rng=rng,
                bad_mask_planes=self.config.bad_mask_planes,
                # Do not fail if the warped exposure does not completely
                # overlap coadd_bbox. This is expected.
                verify=False,
            )
            warp, noise_warp, mfrac_warp, _ = warped_exposures

            # Check that warp has no overlap with non-trivial pixels in warp
            msk = ~(np.isnan(warp.image.array))
            assert (
                (final_warp.variance.array > 0)
                & (~np.isinf(final_warp.variance.array))
                & msk
            ).sum() == 0
            num_good_pixels = msk.sum()

            # Copy the good pixels from warped exposures on to the Warp
            final_warp.image.array[msk] = warp.image.array[msk]
            final_warp.variance.array[msk] = warp.variance.array[msk]
            # NO_DATA will need special handling
            final_warp.mask.array[msk] = warp.mask.array[msk]

            final_noise_warp.image.array[msk] = noise_warp.image.array[msk]
            final_noise_warp.variance.array[msk] = noise_warp.variance.array[msk]
            # NO_DATA will need special handling
            final_noise_warp.mask.array[msk] = noise_warp.mask.array[msk]

            final_mfrac_warp.image.array[msk] = mfrac_warp.image.array[msk]
            final_mfrac_warp.variance.array[msk] = mfrac_warp.variance.array[msk]
            # NO_DATA will need special handling
            final_mfrac_warp.mask.array[msk] = mfrac_warp.mask.array[msk]

            total_good_pixels += num_good_pixels
            inputRecorder.addCalExp(exp, data_id["detector"], num_good_pixels)

            final_warp.setFilter(exp.getFilter())

            np.testing.assert_array_equal(noise_warp.mask.array, warp.mask.array)

            del exp, warp, mfrac_warp
            del noise_warp  # Clear out all noise images from memory

            self.log.debug("Warped %d exposures out of %d", iexp + 1, len(exposures))

        # The warped PSF is not suitable for WL shear measurement.
        # We nevertheless attach it here and recompute the PSF for shear
        # during coaddition.
        inputRecorder.finish(final_warp, total_good_pixels)
        final_psf = CoaddPsf(inputRecorder.coaddInputs.ccds, coadd_wcs)
        final_warp.setPsf(final_psf)

        result = pipeBase.Struct(
            warp=final_warp,
            mfrac_warp=final_mfrac_warp.image,
            noise0_warp=final_noise_warp.getMaskedImage(),
        )
        return result

    def _apply_all_calibrations(
        self,
        exp: Exposure,
        visit_summary: ExposureCatalog | None,
    ) -> None:
        """Apply all of the calibrations from visit_summary to the exposure.

        Specifically, this method updates the following (if available) to the
        input exposure ``exp`` in place from ``visit_summary``:

        - Aperture correction map
        - Photometric calibration
        - PSF
        - WCS

        Parameters
        ----------
        exp : `~lsst.afw.image.Exposure`
            Exposure to be updated.
        visit_summary : `~lsst.afw.table.ExposureCatalog` | None
            Table of visit summary information.  If not None, the visit summary
            information will be used to update the calibration of the input
            exposures. Otherwise, the input exposures will be used as-is.

        Raises
        ------
        RuntimeError
            Raised if ``visit_summary`` is provided but does not contain a
            record corresponding to ``exp``.
        """
        if not visit_summary:
            return

        self.log.debug("Updating calibration from visit summary.")

        detector = exp.info.getDetector().getId()
        row = visit_summary.find(detector)

        if row is None:
            raise RuntimeError(
                f"Unexpectedly incomplete visit_summary: {detector=} is missing."
            )

        if photo_calib := row.getPhotoCalib():
            exp.setPhotoCalib(photo_calib)
        else:
            self.log.warning(
                "No photometric calibration found in visit summary for detector = %s.",
                detector,
            )

        if wcs := row.getWcs():
            exp.setWcs(wcs)
        else:
            self.log.warning(
                "No WCS found in visit summary for detector = %s.", detector
            )

        if psf := row.getPsf():
            exp.setPsf(psf)
        else:
            self.log.warning(
                "No PSF found in visit summary for detector = %s.", detector
            )

        if apcorr_map := row.getApCorrMap():
            exp.setApCorrMap(apcorr_map)
        else:
            self.log.warning(
                "No aperture correction map found in visit summary for detector = %s.",
                detector,
            )

    @classmethod
    def get_seed_from_data_id(cls, data_id) -> int:
        """Get a seed value given a data_id.

        This method generates a unique, reproducible pseudo-random number for
        a data id. This is not affected by ordering of the input, or what
        set of visits, ccds etc. are given.

        This is implemented as a public class method, so that simulations that
        don't necessary deal with the middleware can mock up a ``data_id``
        instance, or orverride this method with a different one to obtain a
        seed value consistent with the pipeline task.

        Parameters
        ----------
        data_id : DataCoordinate
            Data identifier dictionary.

        Returns
        -------
        seed : int
            A unique seed for this data_id to seed a random number generator.
        """
        packer = Instrument.make_default_dimension_packer(data_id, is_exposure=False)
        seed = packer.pack(data_id, returnMaxBits=False)
        return seed
