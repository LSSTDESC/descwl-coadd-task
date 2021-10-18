# see TODO items in code below
from __future__ import annotations
import typing

import numpy as np
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.image as afwImage
from lsst.pipe.tasks.coaddBase import makeSkyInfo
import lsst.utils
import lsst.geom as geom
from lsst.pex.config import Field
from descwl_coadd import make_coadd_obs


class CoaddInCellsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    # not having instrument makes it possible to
    # combine
    # calexp
):
    calExpList = cT.Input(
        doc="Input exposures to be resampled and optionally PSF-matched onto a SkyMap projection/patch",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/box and projection/wcs for coadded exposures",
        name="skyMap",
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    coadd = cT.Output(
        doc="Coadded image",
        name="coaddsInCellsV1",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band", "instrument")
    )
    # Come back and apply jointcal/fgcm later to the calexp s.


class CoaddInCellsConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=CoaddInCellsConnections):
    """Configuration parameters for the `CoaddInCellsTask`.
    """
    seed = Field(
        dtype=int,
        optional=False,
        doc="Base seed for the random number generator",
    )
    interp_bright = Field(
        dtype=bool,
        default=True,
        doc="Interpolate bright stars?"
    )


# TODO Require band argument and only a single band
class CoaddInCellsTask(pipeBase.PipelineTask):
    """Perform coaddition
    """

    ConfigClass = CoaddInCellsConfig
    _DefaultName = "coaddsInCellsV1"

    # @pipeBase.timeMethod
    def run(self,
            calExpList: typing.List[lsst.afw.image.ExposureF],
            skyInfo: pipeBase.Struct) -> pipeBase.Struct:
        # import pdb

        self.log.info("seed: %d", self.config.seed)
        self.log.info("num exp: %d", len(calExpList))

        patchId = skyInfo.patchInfo.getIndex()[1]+5 + skyInfo.patchInfo.getIndex()[0]
        hash_key = hash_function(self.config.seed, skyInfo.tractInfo.getId(),
                patchId, calExpList[0].dataId["band"])
        rng = np.random.RandomState(hash_key)

        # We need to explicitly get the images since we deferred loading.
        # The line below is just an example illustrating this.
        # We should preferably get them sequentially instead of loading all.
        # calExpList = [calexp.get() for calexp in calExpList[:10]]

        # for calexp in calExpList[0:10]:
        #     # import numpy as np
        #     # import esutil as eu
        #     import pdb
        #     self.log.info('band: %s' % calexp.dataId['band'])
        #     calexp = calexp.get()
        #     # m = calexp.mask.array
        #     # v = calexp.variance.array
        #     # w = np.where(m == 0)
        #     #
        #     # eu.stat.print_stats(v.ravel())
        #     # eu.stat.print_stats(v[w].ravel())
        #     #
        #     pdb.set_trace()
        #
        # from IPython import embed; embed()
        # pdb.set_trace()

        # Erin Sheldon has to fill in the interfaces here and
        # replace calExpList[0] with the coadd.
        # coaddedImage = calExpList[0].get()

        data = make_inputs(
            explist=calExpList,
            skyInfo=skyInfo,
            rng=rng,
            num_to_keep=3,
        )
        coadd_obs = make_coadd_obs(
            exps=data["explist"],
            coadd_wcs=data["coadd_wcs"],
            coadd_bbox=data["coadd_bbox"],
            psf_dims=data["psf_dims"],
            rng=rng,
            remove_poisson=True,  # no object poisson noise in sims
        )

        # TODO learn how to save the noise exp as well
        return pipeBase.Struct(coadd=coadd_obs.coadd_exp)

    # @lsst.utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC: pipeBase.ButlerQuantumContext,
                   inputRefs: pipeBase.InputQuantizedConnection,
                   outputRefs: pipeBase.OutputQuantizedConnection):
        """Construct warps and then coadds

        Notes
        -----

        PipelineTask (Gen3) entry point to warp. This method is analogous to
        `runDataRef`. See lsst.pipe.tasks.makeCoaddTempExp.py for comparison.
        """
        # Read in the inputs via the butler
        inputs = butlerQC.get(inputRefs)

        # Process the skyMap to WCS and other useful sky information
        skyMap = inputs.pop("skyMap")  # skyInfo below will contain this skyMap
        quantumDataId = butlerQC.quantum.dataId
        skyInfo = makeSkyInfo(
            skyMap, tractId=quantumDataId["tract"],
            patchId=quantumDataId["patch"],
        )

        # Run the warp and coaddition code
        outputs = self.run(inputs["calExpList"], skyInfo=skyInfo)

        # Persist the results via the butler
        butlerQC.put(outputs.coadd, outputRefs.coadd)


def make_inputs(explist, skyInfo, rng, num_to_keep=None):
    """Make inputs for the coadding code

    Parameters
    ----------
    explist: `list` [`ExposureF`]
        List of exposures to be coadded
    skyInfo: `dict`
        The skyInfo dict, must have .wcs and .bbox
    rng: `~np.random.RandomState`
        Random number generator for noise image generation
    num_to_keep: `int`, optional
        Optionally keep this many exposures

    Returns
    -------
    dict with keys
        'explist': list of exposures to use
        'coadd_wcs': DM wcs object
        'coadd_bbox': DM bbox object
        'psf_dims': dimensions of psf
    """

    bands = set()
    for exp in explist:
        bands.add(exp.dataId["band"])

    if len(bands) > 1:
        raise ValueError(
            "Found %d bands %s, expected one" % (len(bands), bands)
        )

    # TODO Arun add code to remove calexp that have edges
    cell_wcs = skyInfo.wcs
    cell_corners = [cell_wcs.pixelToSky(corner.x, corner.y) for corner in skyInfo.bbox.getCorners()]
    edgeless_explist = []
    for exp in explist:
        calexp_bbox = exp.get(component='bbox')
        calexp_wcs = exp.get(component='wcs')
        if np.all([calexp_bbox.contains(geom.Point2I(calexp_wcs.skyToPixel(corner))) for corner in cell_corners]):
            edgeless_explist.append(exp)

    if num_to_keep is not None:
        ntot = len(edgeless_explist)
        mid = ntot // 4
        edgeless_explist = edgeless_explist[mid:mid + num_to_keep]

    # base psf size on last exp
    psf = edgeless_explist[0].get(component='psf')
    pos = geom.Point2D(x=100, y=100)
    psfim = psf.computeImage(pos)

    psf_dims = psfim.array.shape
    psf_dims = (max(psf_dims), ) * 2

    return {
        "explist": edgeless_explist,
        "coadd_wcs": skyInfo.wcs,
        "coadd_bbox": skyInfo.bbox,
        "psf_dims": psf_dims,
    }


def get_noise_exp(exp, rng):
    """Get a noise image based on the input exposure

    TODO gain correct separately in each amplifier, currently
    averaged

    Parameters
    ----------
    exp: `~lsst.afw.image.ExposureF`
        The exposure upon which to base the noise

    Returns
    -------
    noise_exp: `~lsst.afw.image.ExposureF`
        A noise exposure with the same WCS and size as the input exposure.
    """
    signal = exp.image.array
    variance = exp.variance.array.copy()

    use = np.where(np.isfinite(variance) & np.isfinite(signal))

    gains = [
        amp.getGain() for amp in exp.getDetector().getAmplifiers()
    ]
    mean_gain = np.mean(gains)

    corrected_var = variance[use] - signal[use] / mean_gain

    medvar = np.median(corrected_var)

    noise_image = rng.normal(scale=np.sqrt(medvar), size=signal.shape)

    ny, nx = signal.shape
    nmimage = afwImage.MaskedImageF(width=nx, height=ny)
    assert nmimage.image.array.shape == (ny, nx)

    nmimage.image.array[:, :] = noise_image
    nmimage.variance.array[:, :] = medvar
    nmimage.mask.array[:, :] = exp.mask.array[:, :]

    noise_exp = afwImage.ExposureF(nmimage)
    noise_exp.setPsf(exp.getPsf())
    noise_exp.setWcs(exp.getWcs())
    noise_exp.setFilterLabel(exp.getFilterLabel())
    noise_exp.setDetector(exp.getDetector())

    return noise_exp


def hash_function(seed, tract, patch, band):
    """Generate a hash key given the base seed and metadata
    """
    band_map = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5, "y": 6}
    # Add a linear combination of metadata weighted by prime numbers
    hash_key = seed + 131071*tract + 524287*patch + 8388607*band_map[band]
    return hash_key
