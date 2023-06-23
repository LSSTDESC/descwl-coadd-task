from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

import lsst.afw.math as afwMath
import lsst.geom as geom
import numpy as np
from descwl_coadd import make_coadd_obs, make_warps


from lsst.pex.config import Config, ConfigField, Field, RangeField
from lsst.pipe.base import Task

if TYPE_CHECKING:
    import lsst.afw.image


MAX_MASKFRAC, DEFAULT_INTERP = 0.8, "lanczos5"  # TODO: Import these from descwl_coadd


class WarpConfig(Config):
    """Configuration for the WarpTask"""

    psf_dimensions = Field[int](
        default=21,
        doc="Dimensions of the PSF model",
        check=lambda x: x % 2 == 1,  # Check that psf_dimensions is odd.
    )

    max_maskfrac = RangeField[float](
        default=MAX_MASKFRAC,
        min=0.,
        inclusiveMin=True,
        max=1.,
        inclusiveMax=True,
        doc="Maximum fraction of masked pixels to allow",
    )

    warper_config = ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Warper configuration",
    )

    mfrac_warper_config = ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Warper configuration for mask fraction",
    )

    def setDefaults(self):
        super().setDefaults()
        self.config.warp_config.warpingKernelName = DEFAULT_INTERP
        self.config.mfrac_warp_config.warpingKernelName = "bilinear"


class WarpTask(Task):
    """A task to warp the DESC way."""
    ConfigClass = WarpConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warper_config)
        self.mfrac_warper = afwMath.Warper.fromConfig(self.config.mfrac_warper_config)

    def run(self, exposure, coadd_wcs, coadd_bbox, *args, **kwargs) -> lsst.afw.image.ExposureF:
        returned = make_warps(exposure, coadd_wcs, coadd_bbox, psf_dims=(self.config.psf_dimensions, self.config.psf_dimensions), remove_poisson=False, max_maskfrac=self.config.max_maskfrac)
        warp, noise_warp, psf_warp, mfrac_warp, exp_info = returned
        warp.setPsf(psf_warp)
        # TODO: Stop ignoring the other components
        return warp


class AssembleCellCoaddTask(Task):
    pass
