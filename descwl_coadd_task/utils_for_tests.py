from __future__ import annotations

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image as afw_image
import lsst.skymap as skyMap
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import GaussianPsf
from lsst.daf.butler import DataCoordinate, DimensionUniverse
from lsst.pipe.base import InMemoryDatasetHandle

__all__ = [
    "build_skyMap",
    "construct_geometry",
    "generate_photoCalib",
    "generate_data_id",
    "fill_exposure",
]


PIXEL_SCALE = 0.2
"""Pixel scale in arcseconds"""


def build_skyMap():
    """Build a simple skyMap."""
    simpleMapConfig = skyMap.ringsSkyMap.RingsSkyMapConfig()
    simpleMapConfig.numRings = 120
    simpleMapConfig.pixelScale = 0.2
    simpleMapConfig.projection = "TAN"
    simpleMapConfig.tractBuilder = "cells"
    simpleMapConfig.tractOverlap = 1 / 60

    return skyMap.ringsSkyMap.RingsSkyMap(simpleMapConfig)


def generate_photoCalib(meanCalibration=1e-4, calibrationErr=1e-5):
    """Generate a PhotoCalib instance."""

    return afw_image.PhotoCalib(
        meanCalibration,
        calibrationErr,
    )


def generate_data_id(
    *,
    tract: int = 3828,
    patch: int = 41,
    band: str = "fake",
    detector_id: int = 9,
    visit_id: int = 1234,
    detector_max: int = 189,
    visit_max: int = 1000000,
) -> DataCoordinate:
    """Generate a DataCoordinate instance to use as data_id.

    Parameters
    ----------
    tract : `int`, optional
        Tract ID for the data_id
    patch : `int`, optional
        Patch ID for the data_id
    band : `str`, optional
        Band for the data_id
    detector_id : `int`, optional
        Detector ID for the data_id
    visit_id : `int`, optional
        Visit ID for the data_id
    detector_max : `int`, optional
        Maximum detector ID for the data_id
    visit_max : `int`, optional
        Maximum visit ID for the data_id

    Returns
    -------
    data_id : `lsst.daf.butler.DataCoordinate`
        An expanded data_id instance.
    """
    universe = DimensionUniverse()

    instrument = universe["instrument"]
    instrument_record = instrument.RecordClass(
        name="DummyCam",
        class_name="lsst.obs.base.instrument_tests.DummyCam",
        detector_max=detector_max,
        visit_max=visit_max,
    )

    skymap = universe["skymap"]
    skymap_record = skymap.RecordClass(name="test_skymap")

    band_element = universe["band"]
    band_record = band_element.RecordClass(name=band)

    visit = universe["visit"]
    visit_record = visit.RecordClass(id=visit_id, instrument="test")

    detector = universe["detector"]
    detector_record = detector.RecordClass(id=detector_id, instrument="test")

    physical_filter = universe["physical_filter"]
    physical_filter_record = physical_filter.RecordClass(
        name=band, instrument="test", band=band
    )

    patch_element = universe["patch"]
    patch_record = patch_element.RecordClass(
        skymap="test_skymap",
        tract=tract,
        patch=patch,
    )

    if "day_obs" in universe:
        day_obs_element = universe["day_obs"]
        day_obs_record = day_obs_element.RecordClass(id=20240201, instrument="test")
    else:
        day_obs_record = None

    # A dictionary with all the relevant records.
    record = {
        "instrument": instrument_record,
        "visit": visit_record,
        "detector": detector_record,
        "patch": patch_record,
        "tract": tract,
        "band": band_record.name,
        "skymap": skymap_record.name,
        "physical_filter": physical_filter_record,
    }

    if day_obs_record:
        record["day_obs"] = day_obs_record

    # A dictionary with all the relevant recordIds.
    record_id = record.copy()
    for key in ("visit", "detector"):
        record_id[key] = record_id[key].id

    data_id = DataCoordinate.standardize(record_id, universe=universe)
    return data_id.expanded(record)


def fill_exposure(exposure, rng, mask_pixel=False, mask_bitname="SAT"):
    """Fill an exposure with random data.

    Parameters
    ----------
    mask_pixel : `bool`, optional
        If True, a pixel will be masked in the exposure.
    mask_bitname : `str`, optional
        Name of the mask bit to set.

    Returns
    -------
    exposure : `lsst.afw.image.ExposureF`
        The modified exposure.

    Notes
    -----
    The input ``exposure`` is modified in-place.
    """
    exposure.maskedImage.image.array = (
        rng.uniform(size=exposure.image.array.shape).astype(np.float32) * 1000
    )
    exposure.maskedImage.variance.array = rng.uniform(
        low=0.98,
        high=1.0,
        size=exposure.image.array.shape,
    ).astype(np.float32)

    if mask_pixel:
        exposure.maskedImage.mask[5, 5] = afw_image.Mask.getPlaneBitMask(mask_bitname)

    return exposure


def construct_geometry(visit_id: int, nx: int = 4072, ny: int = 4000):
    """Generate the necessary dataRefs to run the MakeWarpTask.

    This is primarily setting up the geometry of the input exposures,
    along with some metadata.
    """

    match visit_id:

        case 204706:
            crpixList = [
                lsst.geom.Point2D(1990.63, 2018.46),
                lsst.geom.Point2D(2186.46, 2044.82),
            ]
            crvalList = [
                lsst.geom.SpherePoint(
                    57.14 * lsst.geom.degrees, -36.51 * lsst.geom.degrees
                ),
                lsst.geom.SpherePoint(
                    57.39 * lsst.geom.degrees, -36.63 * lsst.geom.degrees
                ),
            ]
            cdMatrix = lsst.afw.geom.makeCdMatrix(
                scale=PIXEL_SCALE * lsst.geom.arcseconds,
                orientation=122.9 * lsst.geom.degrees,
            )
            detectorNums = [158, 161]

        case 204708:
            crpixList = [
                lsst.geom.Point2D(2111.48, 1816.14),
                lsst.geom.Point2D(1971.88, 1899.54),
                lsst.geom.Point2D(1863.09, 1939.27),
            ]
            crvalList = [
                lsst.geom.SpherePoint(
                    57.28 * lsst.geom.degrees, -36.66 * lsst.geom.degrees
                ),
                lsst.geom.SpherePoint(
                    57.19 * lsst.geom.degrees, -36.35 * lsst.geom.degrees
                ),
                lsst.geom.SpherePoint(
                    57.43 * lsst.geom.degrees, -36.48 * lsst.geom.degrees
                ),
            ]
            cdMatrix = lsst.afw.geom.makeCdMatrix(
                scale=PIXEL_SCALE * lsst.geom.arcseconds,
                orientation=121.52 * lsst.geom.degrees,
            )
            detectorNums = [32, 36, 39]

        case 174534:
            crpixList = [
                lsst.geom.Point2D(1929.4499812439476, 2147.1179041443015),
                lsst.geom.Point2D(2012.9460541750886, 2073.7299107964782),
                lsst.geom.Point2D(2259.5002281485358, 2105.9109483821226),
            ]
            crvalList = [
                lsst.geom.SpherePoint(
                    55.94990987622795 * lsst.geom.degrees,
                    -36.18810977436167 * lsst.geom.degrees,
                ),
                lsst.geom.SpherePoint(
                    56.02684930702103 * lsst.geom.degrees,
                    -35.96612774405456 * lsst.geom.degrees,
                ),
                lsst.geom.SpherePoint(
                    55.733500259251784 * lsst.geom.degrees,
                    -35.89282833381862 * lsst.geom.degrees,
                ),
            ]
            cdMatrix = lsst.afw.geom.makeCdMatrix(
                scale=PIXEL_SCALE * lsst.geom.arcseconds,
                orientation=16.69 * lsst.geom.degrees,
            )
            detectorNums = [12, 15, 16]

        case 397330:

            crpixList = [
                lsst.geom.Point2D(2038.0123728146252, 1886.2009723798765),
                lsst.geom.Point2D(2165.4630755512139, 1991.9349231803017),
            ]
            crvalList = [
                lsst.geom.SpherePoint(
                    56.05982507585136 * lsst.geom.degrees,
                    -36.04171236047266 * lsst.geom.degrees,
                ),
                lsst.geom.SpherePoint(
                    56.01232760335289 * lsst.geom.degrees,
                    -35.80314141320685 * lsst.geom.degrees,
                ),
            ]
            cdMatrix = lsst.afw.geom.makeCdMatrix(
                scale=PIXEL_SCALE * lsst.geom.arcseconds,
                orientation=352.56 * lsst.geom.degrees,
            )

            detectorNums = [78, 117]

        case 1248:
            crpixList = [
                lsst.geom.Point2D(2047.84, 1909.10),
                lsst.geom.Point2D(2110.05, 1982.96),
                lsst.geom.Point2D(2104.22, 1780.364),
            ]
            crvalList = [
                lsst.geom.SpherePoint(
                    56.16 * lsst.geom.degrees, -36.84 * lsst.geom.degrees
                ),
                lsst.geom.SpherePoint(
                    56.33 * lsst.geom.degrees, -36.64 * lsst.geom.degrees
                ),
                lsst.geom.SpherePoint(
                    56.40 * lsst.geom.degrees, -36.96 * lsst.geom.degrees
                ),
            ]
            cdMatrix = lsst.afw.geom.makeCdMatrix(
                scale=PIXEL_SCALE * lsst.geom.arcseconds,
                orientation=123.5 * lsst.geom.degrees,
            )
            detectorNums = [107, 114, 146]

        case _:
            raise ValueError(
                f"{visit_id} is not supported. It must be one of [1248, 204706, 397330]"
            )

    dataRefs = []

    for detId, detectorNum in enumerate(detectorNums):
        wcs = lsst.afw.geom.makeSkyWcs(crpixList[detId], crvalList[detId], cdMatrix)
        exposure = afw_image.ExposureF(nx, ny)

        # Set the PhotoCalib, Wcs and detector objects of this exposure.
        exposure.setPhotoCalib(generate_photoCalib())
        exposure.setPsf(GaussianPsf(25, 25, 2.5))
        exposure.setFilter(afw_image.FilterLabel(physical="fakeFilter", band="fake"))
        exposure.setWcs(wcs)

        detectorName = f"detector {detectorNum}"
        detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(
            name=detectorName,
            id=detectorNum,
        ).detector
        exposure.setDetector(detector)

        dataId_dict = {"detector_id": detectorNum, "visit_id": visit_id, "band": "fake"}
        dataId = generate_data_id(**dataId_dict)
        dataRef = InMemoryDatasetHandle(exposure, dataId=dataId)

        dataRefs.append(dataRef)

    return dataRefs
