from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.pipe.base as pipeBase
from descwl_coadd import (
    DEFAULT_INTERP,
    FLAGS2INTERP,
    MAX_MASKFRAC,
    get_bad_mask,
    get_info_struct,
    make_coadd,
    make_stacker,
)
from lsst.cell_coadds import (
    CellIdentifiers,
    CoaddUnits,
    CommonComponents,
    GridContainer,
    MultipleCellCoadd,
    OwnedImagePlanes,
    PatchIdentifiers,
    SingleCellCoadd,
    UniformGrid,
)
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.pex.config import ConfigField, ConfigurableField, Field, ListField, RangeField
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.pipe.tasks.makeWarp import reorderRefs
from lsst.pipe.tasks.scaleZeroPoint import ScaleZeroPointTask
from lsst.skymap import BaseSkyMap

log = logging.getLogger(__name__)


class AssembleShearCoaddConnections(
    PipelineTaskConnections,
    dimensions=(
        "tract",
        "patch",
        "band",
        "skymap",
    ),
    defaultTemplates={"coaddType": ""},
):
    input_warps = Input(
        doc="Input warped exposures to be coadded",
        name="calexp_descwarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    mfrac_warps = Input(
        doc="Masked fraction on the input warp",
        # name="calexp_mfracwarp",
        name="calexp_mfrac_descwarp",
        storageClass="ImageF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    visitSummary = Input(
        doc="Input visit-summary catalog with updated calibration objects.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=(
            "instrument",
            "visit",
        ),
        deferLoad=True,
        multiple=True,
    )
    skyMap = Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded "
        "exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    multiple_cell_coadd = Output(
        doc="Output coadded exposure, produced by stacking input warps",
        name="shearCoadd",
        storageClass="MultipleCellCoadd",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)  # Harmless and a no-op for now.
        if not config:
            return

        # Dynamically set input connections for noise images depending on the
        # number of noise realizations specified in the config.
        for n in range(config.num_noise_realizations):
            noise_warps = Input(
                doc="Input noise warp to be coadded.",
                name=f"{config.connections.coaddType}calexp_noise{n}_descwarp",
                storageClass="ImageF",
                dimensions=("tract", "patch", "skymap", "visit", "instrument"),
                deferLoad=True,
                multiple=True,
            )
            setattr(self, f"noise{n}_warps", noise_warps)


class AssembleShearCoaddConfig(
    PipelineTaskConfig, pipelineConnections=AssembleShearCoaddConnections
):
    """Configuration for AssembleShearCoaddTask."""

    bad_mask_planes = ListField[str](
        doc="Mask planes that, if set, the corresponding pixel would not be "
        "included in the coadd.",
        default=FLAGS2INTERP,
    )
    psf_dimensions = Field[int](
        default=21,
        doc="Dimensions of the PSF image stamp size to be assigned to cells.",
        # Check that psf_dimensions is positive and odd.
        check=lambda x: (x > 0) and (x % 2 == 1),
    )
    psf_warper = ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Configuration for the warper that warps the PSF."
        "This must match the configuration used to warp the images.",
    )
    max_maskfrac = RangeField[float](
        default=0.9,
        min=0.0,
        inclusiveMin=True,
        max=1.0,
        inclusiveMax=True,
        doc="Maximum fraction of masked pixels to allow. Warps with maskfrac "
        "values greater than or equal to this value within the cell's "
        "bounding box (outer) are excluded from the coadd. "
        "Values must be in the range [0, 1].",
    )
    num_noise_realizations = Field[int](
        default=1,
        doc=(
            "Number of noise planes to include in the coadd. "
            "This should not exceed the corresponding config parameter "
            "specified in `MakeShearWarpConfig`. "
            "Currently, the only values accepted are 0 and 1."
        ),
        # The upper limit of 1 is checked in `validate` method for now.
        check=lambda x: x >= 0,  # Check that num_noise_realizations is non-negative.
    )
    remove_poisson = Field[bool](
        doc="Remove Poisson noise from the variance plane?"
        "Currently, this is not yet supported.",
        default=False,
    )
    do_scale_zero_point = Field[bool](
        doc="Scale the warps to a common zero point? This requires the input "
        "warps to have a PhotoCalib object attached to them.",
        default=False,
    )
    scale_zero_point = ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to scale warps to a common zero point",
    )

    def setDefaults(self):
        super().setDefaults()
        # Set default values from the descwl_coadd package here.
        # These override the default values set in the config definition.
        self.max_maskfrac = MAX_MASKFRAC
        self.bad_mask_planes = FLAGS2INTERP
        # Configure the warpers to have the same values as in descwl_coadd.
        self.psf_warper.warpingKernelName = DEFAULT_INTERP

    def validate(self):
        # These validations will be removed as features are expanded.
        if self.remove_poisson:
            raise ValueError("remove_poisson is not supported yet.")
        if self.num_noise_realizations > 1:
            raise ValueError("num_noise_realizations > 1 is not supported yet.")
        super().validate()


class AssembleShearCoaddTask(PipelineTask):
    ConfigClass = AssembleShearCoaddConfig
    _DefaultName = "assembleShearCoadd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf_warper = afwMath.Warper.fromConfig(self.config.psf_warper)
        self.makeSubtask("scale_zero_point")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        visitOrder = [ref.datasetRef.dataId["visit"] for ref in inputRefs.visitSummary]
        visitOrder.sort()
        inputRefs = reorderRefs(inputRefs, visitOrder, dataIdKey="visit")
        inputData = butlerQC.get(inputRefs)

        if not inputData["input_warps"]:
            raise NoWorkFound("No input warps provided for co-addition")
        self.log.info("Found %d input warps", len(inputData["input_warps"]))
        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case _makeSupplementaryData
        # needs it
        skyMap = inputData["skyMap"]
        outputDataId = butlerQC.quantum.dataId

        inputData["skyInfo"] = makeSkyInfo(
            skyMap, tractId=outputDataId["tract"], patchId=outputDataId["patch"]
        )

        self.common = CommonComponents(
            units=CoaddUnits.nJy,
            wcs=inputData["skyInfo"].patchInfo.wcs,
            band=outputDataId.get("band", None),
            identifiers=PatchIdentifiers.from_data_id(outputDataId),
        )

        returnStruct = self.run(
            input_warps=inputData["input_warps"],
            noise_warps=inputData["noise0_warps"],
            mfrac_warps=inputData["mfrac_warps"],
            skyInfo=inputData["skyInfo"],
        )
        butlerQC.put(returnStruct, outputRefs)
        return returnStruct

    def run(self, *, input_warps, mfrac_warps, noise_warps, skyInfo):
        raise NotImplementedError("This method is not yet implemented.")

    @staticmethod
    def _construct_grid(skyInfo):
        # grid has no notion about border or inner/outer boundaries.
        # So we have to clip the outermost border when constructing the grid.
        grid_bbox = skyInfo.patchInfo.outer_bbox.erodedBy(
            skyInfo.patchInfo.getCellBorder()
        )
        grid = UniformGrid.from_bbox_cell_size(
            grid_bbox, skyInfo.patchInfo.getCellInnerDimensions()
        )
        return grid

    def _construct_grid_container(self, skyInfo, statsCtrl):
        """Construct a grid of AccumulatorMeanStack instances.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            A Struct object

        Returns
        -------
        gc : `~lsst.cell_coadds.GridContainer`
            A GridContainer object container one AccumulatorMeanStack per cell.
        """
        grid = self._construct_grid(skyInfo)

        # Initialize the grid container with AccumulatorMeanStacks
        gc = GridContainer[AccumulatorMeanStack](grid.shape)
        for cellInfo in skyInfo.patchInfo:
            coadd_dims = (
                cellInfo.outer_bbox.getDimensions().x,
                cellInfo.outer_bbox.getDimensions().y,
            )
            # Note: make_stacker generates a new instance of StatisticsControl
            # if not provided with one, and hardcodes parameters that should be
            # configurable eventually.
            # TODO: Make this more configurable.
            stacker = make_stacker(coadd_dims, statsCtrl)
            gc[cellInfo.index] = stacker

        return gc

    def _construct_stats_control(self):
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.bad_mask_planes))
        statsCtrl.setNanSafe(True)
        return statsCtrl


@dataclass
class PackedExposure:
    """A dataclass to hold the various warpes.

    This class is used to package the various types of warps to pass to the
    `make_coadd` function from the `descwl_coadd` package.

    Parameters
    ----------
    warp : `~lsst.afw.image.ExposureF`
        The warped exposure.
    noise_warp : `~lsst.afw.image.ExposureF`
        The noise warped exposure.
    mfrac_warp : `~lsst.afw.image.ExposureF`
        The masked fraction warped exposure.
    exp_info : `dict`
        A dictionary containing the exposure metadata.
    """

    warp: afwImage.ExposureF
    noise_warp: afwImage.ExposureF
    mfrac_warp: afwImage.ExposureF
    exp_info: dict[str, Any]

    # Mimimal set of methods to make this duck-type as Exposure objects.
    def getFilter(self):
        return self.warp.getFilter()


class AssembleShearCoaddSlowConnections(AssembleShearCoaddConnections):
    pass


class AssembleShearCoaddSlowConfig(
    AssembleShearCoaddConfig, pipelineConnections=AssembleShearCoaddSlowConnections
):
    """Configuration for AssembleShearCoaddSlowTask.

    This is a trivial sub-class of `AssembleShearCoaddConfig`.
    """


class AssembleShearCoaddSlowTask(AssembleShearCoaddTask):
    """A PipelineTask that assembles coadds from warps, one cell at a time.

    The run method of this PipelineTask calls the `make_coadd` function from
    the `descwl_coadd` package. It constructs one cell completely before moving
    onto the next one, and thus leads to I/O patterns that are too suboptimal
    to be used for large-scale production runs. This task is used to check for
    consistency with `AssembleShearCoaddTask` which uses the barebones from the
    `descwl_coadd` package.
    """

    ConfigClass = AssembleShearCoaddSlowConfig
    _DefaultName = "assembleShearCoaddSlow"

    def run(self, *, input_warps, noise_warps, mfrac_warps, skyInfo):
        """Coadd a set of warped images, along with noise and masked fractions.

        Parameters
        ----------
        input_warps : `Iterable` [`~lsst.daf.butler.DeferredDatasetHandle`]
            List of data references to warped images.
        noise_warps : `Iterable` [`~lsst.daf.butler.DeferredDatasetHandle`]
            List of data references of warped noise images.
        mfrac_warps : `Iterable` [`~lsst.daf.butler.DeferredDatasetHandle`]
            List of data references of masked fraction images.
        skyInfo : `~lsst.pipe.base.Struct`
            A Struct object

        Returns
        -------
        coaddExposure : `~lsst.afw.image.ExposureF`
            The coadded exposure.
        """
        # statsCtrl = self._construct_stats_control()

        # gc = self._construct_grid_container(skyInfo, statsCtrl)
        # coadd_inputs_gc = GridContainer(gc.shape)
        edge = afwImage.Mask.getPlaneBitMask("EDGE")

        cells: list[SingleCellCoadd] = []
        for cellInfo in skyInfo.patchInfo:
            # coadd_inputs = self.inputRecorder.makeCoaddInputs()
            # # Reserve the absolute maximum of how many ccds, visits
            # # we could potentially have.
            # coadd_inputs.ccds.reserve(len(input_warps))
            # coadd_inputs.visits.reserve(len(input_warps))
            # coadd_inputs_gc[cellInfo.index] = coadd_inputs

            bbox = cellInfo.outer_bbox
            # Read in one warp at a time, and accumulate it in all the cells
            # that it completely overlaps.

            # Ideally, we should pass these components as a lazy iterator.
            # However, the pbar in make_coadd needs it as a list to
            # measure the length.
            exps, psfs, wcss = [], [], []

            for warpRef, noiseWarpRef, mfracWarpRef in zip(
                input_warps,
                noise_warps,
                mfrac_warps,
                strict=True,
            ):
                warp = warpRef.get(parameters={"bbox": bbox})
                mi = warp.getMaskedImage()

                # Coadd the warp onto the cells it completely overlaps.
                if (mi.getMask().array & edge).any():
                    self.log.debug(
                        "Skipping %s in cell %s because it has an EDGE",
                        warpRef.dataId,
                        cellInfo.index,
                    )
                    continue

                _, maskfrac = get_bad_mask(mi, self.config.bad_mask_planes)
                # We do not skip processing the warp based on maskfrac at this
                # point. We let make_coadd handle it.

                # Convert the noise warp to an Exposure object.
                noise_warp = afwImage.ExposureF(
                    maskedImage=noiseWarpRef.get(parameters={"bbox": bbox}),
                    wcs=warp.wcs,
                )

                # Pre-process the warp before coadding.
                if self.config.do_scale_zero_point:
                    self.scale_zero_point.run(exposure=warp, dataRef=warp)
                    self.scale_zero_point.run(exposure=noise_warp, dataRef=noiseWarpRef)

                # Convert the noise warp to an Exposure object.
                mfrac_warp = afwImage.ExposureF(warp, deep=True)
                mfrac_warp.image = mfracWarpRef.get(parameters={"bbox": bbox})
                # TODO: Remove accessing the image attr on the right.

                center = skyInfo.wcs.pixelToSky(geom.Point2D(bbox.getCenter()))
                subcat = warp.getInfo().getCoaddInputs().ccds.subsetContaining(center)

                if not len(subcat) == 1:
                    self.log.warning("Found %d ccds in %s", len(subcat), warpRef.dataId)
                    continue

                row = subcat[0]

                psf = row.getPsf()
                wcs = row.getWcs()

                exp_info = get_info_struct(1)
                exp_info["exp_id"] = row.getId()
                exp_info["weight"] = row["weight"]
                exp_info["maskfrac"] = maskfrac

                packed_exp = PackedExposure(warp, noise_warp, mfrac_warp, exp_info)
                exps.append(packed_exp)
                psfs.append(psf)
                wcss.append(wcs)

            if not exps:
                continue

            # Since noise warps are already generated, make_coadd does not
            # need a random number generator anymore.
            rng = None

            coadd_data = make_coadd(
                exps=exps,
                coadd_wcs=skyInfo.patchInfo.wcs,
                coadd_bbox=bbox,
                psf_dims=(self.config.psf_dimensions, self.config.psf_dimensions),
                rng=rng,
                remove_poisson=self.config.remove_poisson,
                psfs=psfs,
                wcss=wcss,
                max_maskfrac=self.config.max_maskfrac,
                is_warps=True,
                warper=self.psf_warper,
            )

            image_planes = OwnedImagePlanes(
                image=coadd_data["coadd_exp"].image,
                mask=coadd_data["coadd_exp"].mask,
                variance=coadd_data["coadd_exp"].variance,
                noise_realizations=[coadd_data["coadd_noise_exp"].image],
                mask_fractions=coadd_data["coadd_mfrac_exp"].image,
            )

            identifiers = CellIdentifiers(
                cell=cellInfo.index,
                skymap=self.common.identifiers.skymap,
                tract=self.common.identifiers.tract,
                patch=self.common.identifiers.patch,
                band=self.common.identifiers.band,
            )

            scc = SingleCellCoadd(
                outer=image_planes,
                psf=coadd_data["coadd_psf_exp"].image,
                inner_bbox=cellInfo.inner_bbox,
                inputs=None,
                common=self.common,
                identifiers=identifiers,
            )

            cells.append(scc)
            break

        grid = self._construct_grid(skyInfo)
        mcc = MultipleCellCoadd(
            cells,
            grid=grid,
            outer_cell_size=cellInfo.outer_bbox.getDimensions(),
            inner_bbox=None,
            common=self.common,
            psf_image_size=cells[0].psf_image.getDimensions(),
        )

        return pipeBase.Struct(multiple_cell_coadd=mcc)
