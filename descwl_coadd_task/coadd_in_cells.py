from __future__ import annotations

from typing import Mapping, Tuple

import lsst.geom as geom
import lsst.pipe.base as pipeBase
import numpy as np
from descwl_coadd import make_coadd_obs
from lsst.cell_coadds import (
    ObservationIdentifiers,
    OwnedImagePlanes,
    SingleCellCoaddBuilderConfig,
    SingleCellCoaddBuilderTask,
    singleCellCoaddBuilderRegistry,
)
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pex.config import Field, registerConfigurable
from lsst.skymap import CellInfo


class SCCBuilderConfig(SingleCellCoaddBuilderConfig):

    seed = Field[int](
        doc="Base seed for the random number generator",
        dtype=int,
        optional=False,
    )


@registerConfigurable("descCoaddBuilder", singleCellCoaddBuilderRegistry)
class SCCBuilder(SingleCellCoaddBuilderTask):
    """A concrete class to build single cell coadds"""

    ConfigClass = SCCBuilderConfig

    def run(
        self,
        inputs: Mapping[
            ObservationIdentifiers, Tuple[DeferredDatasetHandle, geom.Box2I]
        ],
        cellInfo: CellInfo,
    ):
        exp_list = [_v[0].get(bbox=_v[1]) for _k, _v in inputs.items()]
        # Any further selection/rejection of exposures should be done here.
        coadd_obs, exp_info = make_coadd_obs(
            exps=exp_list,
            coadd_wcs=cellInfo.wcs,
            coadd_bbox=cellInfo.outer_bbox,
            psf_dims=(self.config.psf_dimensions, self.config.psf_dimensions),
            rng=np.random.RandomState(self.config.seed),
            remove_poisson=True,
        )

        center = coadd_obs.coadd_exp.psf.getAveragePosition()
        image_planes = OwnedImagePlanes(
            image=coadd_obs.coadd_exp.image,
            mask=coadd_obs.coadd_exp.mask,
            variance=coadd_obs.coadd_exp.variance,
            noise_realizations=(coadd_obs.coadd_noise_exp.image,),
            mask_fractions={},
        )
        return pipeBase.Struct(
            image_planes=image_planes,
            psf=coadd_obs.coadd_exp.psf.computeKernelImage(center),
            inputs=inputs,
        )
