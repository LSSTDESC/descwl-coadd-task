from __future__ import annotations

import unittest

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image as afw_image
import lsst.skymap as skyMap
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import GaussianPsf
from lsst.daf.butler import DataCoordinate, DimensionUniverse
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.coaddBase import makeSkyInfo

from descwl_coadd_task import MakeShearWarpConfig, MakeShearWarpTask


class MakeWarpTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_makeWarpSmoke(self):
        """Test basic MakeDirectWarpTask."""

        config = MakeShearWarpConfig()

        data = _make_data(rng=self.rng)
        dataRef = data["dataRef"]
        skyInfo = data["skyInfo"]

        makeWarp = MakeShearWarpTask(config=config)
        inputs = {"calexp_list": [dataRef]}
        result = makeWarp.run(
            inputs["calexp_list"], skyInfo=skyInfo, visit_summary=None
        )

        warp = result.warp
        mfrac = result.mfrac_warp
        noise = result.noise0_warp

        # Ensure we got an exposure out
        self.assertIsInstance(warp, afw_image.ExposureF)
        # Ensure that masked fraction is an ImageF object.
        self.assertIsInstance(mfrac, afw_image.ImageF)
        # Ensure that the noise image is a MaskedImageF object.
        self.assertIsInstance(noise, afw_image.MaskedImageF)
        # Ensure the warp has valid pixels
        self.assertGreater(np.isfinite(warp.image.array.ravel()).sum(), 0)
        # Ensure the warp has the correct WCS
        self.assertEqual(warp.getWcs(), skyInfo.wcs)
        # Ensure that mfrac has pixels between 0 and 1
        self.assertTrue(np.nanmax(mfrac.array) <= 1)
        self.assertTrue(np.nanmin(mfrac.array) >= 0)

        coadd_bbox = skyInfo.bbox
        expected_shape = (coadd_bbox.getHeight(), coadd_bbox.getWidth())
        self.assertEqual(warp.image.array.shape, expected_shape)
        self.assertEqual(mfrac.array.shape, expected_shape)
        self.assertEqual(noise.image.array.shape, expected_shape)

        # we didn't mask any pixels for this test
        self.assertTrue(np.all(mfrac.array[:, :] == 0))

        # TODO add smoke tests of PSF when psf warps are added

    @lsst.utils.tests.methodParameters(mask_bitname=["BAD", "CR", "SAT"])
    def test_makeWarpMfrac(self, mask_bitname):
        """Test mfrac is being properly propagated"""

        config = MakeShearWarpConfig()
        nexpected = 36

        # this masks a single pixel, which then propagates to mfrac
        # in a set of neighboring pixels
        data = _make_data(
            rng=self.rng,
            mask_pixel=True,
            mask_bitname=mask_bitname,
        )
        dataRef = data["dataRef"]
        skyInfo = data["skyInfo"]

        makeWarp = MakeShearWarpTask(config=config)
        inputs = {"calexp_list": [dataRef]}
        result = makeWarp.run(
            inputs["calexp_list"], skyInfo=skyInfo, visit_summary=None
        )

        mfrac = result.mfrac_warp
        warp = result.warp

        wbad = np.where(mfrac.array != 0)
        assert wbad[0].size == nexpected

        iflag = afw_image.Mask.getPlaneBitMask("INTRP")
        wintrp = np.where(warp.mask.array & iflag != 0)
        assert wintrp[0].size == nexpected


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def _make_data(rng, mask_pixel=False, mask_bitname="SAT"):
    ny = 150
    nx = 100

    data = {}

    meanCalibration = 1e-4
    calibrationErr = 1e-5
    data["exposurePhotoCalib"] = afw_image.PhotoCalib(meanCalibration, calibrationErr)
    # An external photoCalib calibration to return
    data["externalPhotoCalib"] = afw_image.PhotoCalib(1e-6, 1e-8)

    crpix = lsst.geom.Point2D(0, 0)
    crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
    cdMatrix = lsst.afw.geom.makeCdMatrix(scale=1.0 * lsst.geom.arcseconds)
    data["skyWcs"] = lsst.afw.geom.makeSkyWcs(crpix, crval, cdMatrix)
    externalCdMatrix = lsst.afw.geom.makeCdMatrix(scale=0.9 * lsst.geom.arcseconds)
    # An external skyWcs to return
    data["externalSkyWcs"] = lsst.afw.geom.makeSkyWcs(crpix, crval, externalCdMatrix)

    exposure = afw_image.ExposureF(nx, ny)
    data["exposure"] = exposure

    exposure.maskedImage.image.array = (
        rng.uniform(size=(ny, nx)).astype(np.float32) * 1000
    )
    exposure.maskedImage.variance.array = rng.uniform(
        low=0.98, high=1.0, size=(ny, nx)
    ).astype(np.float32)

    if mask_pixel:
        exposure.maskedImage.mask[5, 5] = afw_image.Mask.getPlaneBitMask(mask_bitname)

    # set the PhotoCalib and Wcs objects of this exposure.
    exposure.setPhotoCalib(afw_image.PhotoCalib(meanCalibration, calibrationErr))
    exposure.setWcs(data["skyWcs"])
    exposure.setPsf(GaussianPsf(5, 5, 2.5))
    exposure.setFilter(afw_image.FilterLabel(physical="fakeFilter", band="fake"))

    data["visit"] = 100
    data["detector"] = 5
    detectorName = f"detector {data['detector']}"
    detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(
        name=detectorName, id=data["detector"]
    ).detector
    exposure.setDetector(detector)

    dataId_dict = {"detector_id": data["detector"], "visit_id": 1248, "band": "i"}
    dataId = _generate_data_id(**dataId_dict)
    data["dataRef"] = InMemoryDatasetHandle(exposure, dataId=dataId)
    simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
    simpleMapConfig.raList = [crval.getRa().asDegrees()]
    simpleMapConfig.decList = [crval.getDec().asDegrees()]
    simpleMapConfig.radiusList = [0.1]

    data["simpleMap"] = skyMap.DiscreteSkyMap(simpleMapConfig)
    data["tractId"] = 0
    data["patchId"] = data["simpleMap"][0].findPatch(crval).sequential_index
    data["skyInfo"] = makeSkyInfo(data["simpleMap"], data["tractId"], data["patchId"])

    return data


def _generate_data_id(
    *,
    tract: int = 9813,
    patch: int = 42,
    band: str = "r",
    detector_id: int = 9,
    visit_id: int = 1234,
    detector_max: int = 109,
    visit_max: int = 10000,
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
        "tract": 9813,
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

    # TODO: Catching mypy failures on Github Actions should be made easier,
    # perhaps in DM-36873. Igroring these for now.
    data_id = DataCoordinate.standardize(record_id, universe=universe)
    return data_id.expanded(record)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
