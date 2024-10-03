from __future__ import annotations

import unittest

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image as afw_image
import lsst.skymap as skyMap
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import GaussianPsf
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.coaddBase import makeSkyInfo

from descwl_coadd_task import MakeShearWarpConfig, MakeShearWarpTask
from descwl_coadd_task.utils_for_tests import (
    construct_geometry,
    fill_exposure,
    generate_data_id,
    generate_photoCalib,
)

PIXEL_SCALE = 0.2
"""Pixel scale in arcseconds"""


# Use a local skyMap builder to get a DiscreteSkyMap.
def _build_skyMap():
    """Build a simple skyMap."""
    crval = lsst.geom.SpherePoint(
        56.65 * lsst.geom.degrees,
        -36.45 * lsst.geom.degrees,
    )

    simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
    simpleMapConfig.raList = [crval.getRa().asDegrees()]
    simpleMapConfig.decList = [crval.getDec().asDegrees()]
    simpleMapConfig.radiusList = [0.1]
    simpleMapConfig.pixelScale = PIXEL_SCALE

    return skyMap.DiscreteSkyMap(simpleMapConfig)


class MakeWarpTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        # Docstring inherited.
        # This method contains the setup that needs to be done only once for
        # all the tests.
        cls.ny = 4000
        cls.nx = 4072

        cls.skyMap = _build_skyMap()
        cls.photoCalib = generate_photoCalib()
        cls.skyInfo = makeSkyInfo(cls.skyMap, 0, 36)

        # Setup the metadata and empty exposures for warping.
        cls.dataRefs = construct_geometry(visit_id=1248)

    def setUp(self):
        # Docstring inherited.
        # This method contains the setup that needs to be repeated for each
        # test.
        self.rng = np.random.default_rng(12345)

        # Fill the input exposures with random data.
        # This needs to be redone for each test, since the test may have
        # modified the pixel values in-place.
        for exposureRef in self.dataRefs:
            fill_exposure(exposureRef.get(), rng=self.rng)

    def tearDown(self):
        # Docstring inherited.
        # This method cleans up the mask planes before the next test.
        for exposureRef in self.dataRefs:
            exposureRef.get().mask.clearAllMaskPlanes()

    @classmethod
    def _make_data(cls):
        """Generate the necessary dataRefs to run the MakeWarpTask.

        This is primarily setting up the geometry of the input exposures,
        along with some metadata.
        """

        # Calexp WCSs
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

        cls.dataRefs = []

        for detId, detectorNum in enumerate([107, 114, 146]):
            wcs = lsst.afw.geom.makeSkyWcs(crpixList[detId], crvalList[detId], cdMatrix)
            exposure = afw_image.ExposureF(cls.nx, cls.ny)

            # Set the PhotoCalib, Wcs and detector objects of this exposure.
            exposure.setPhotoCalib(cls.photoCalib)
            exposure.setPsf(GaussianPsf(5, 5, 2.5))
            exposure.setFilter(
                afw_image.FilterLabel(physical="fakeFilter", band="fake")
            )
            exposure.setWcs(wcs)

            detectorName = f"detector {detectorNum}"
            detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(
                name=detectorName,
                id=detectorNum,
            ).detector
            exposure.setDetector(detector)

            dataId_dict = {"detector_id": detectorNum, "visit_id": 1248, "band": "fake"}
            dataId = generate_data_id(**dataId_dict)
            dataRef = InMemoryDatasetHandle(exposure, dataId=dataId)

            cls.dataRefs.append(dataRef)

    def test_makeWarpSmoke(self):
        """Test basic MakeDirectWarpTask and check that the ."""

        config = MakeShearWarpConfig()

        makeWarp = MakeShearWarpTask(config=config)

        result = makeWarp.run(
            self.dataRefs,
            skyInfo=self.skyInfo,
            visit_summary=None,
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
        self.assertEqual(warp.getWcs(), self.skyInfo.wcs)
        # Ensure that mfrac has pixels between 0 and 1
        self.assertTrue(np.nanmax(mfrac.array) <= 1)
        self.assertTrue(np.nanmin(mfrac.array) >= 0)

        coadd_bbox = self.skyInfo.bbox
        expected_shape = (coadd_bbox.getHeight(), coadd_bbox.getWidth())
        self.assertEqual(warp.image.array.shape, expected_shape)
        self.assertEqual(mfrac.array.shape, expected_shape)
        self.assertEqual(noise.image.array.shape, expected_shape)

        # We didn't mask any pixels for this test, so the mask bits should be 0
        # except where there is NO_DATA and EDGE bits set due to chip gaps.
        bit_mask = afw_image.Mask.getPlaneBitMask(
            "NO_DATA"
        ) | afw_image.Mask.getPlaneBitMask("EDGE")
        # Slicing mfrac.array can consume too much memory in GitHub Actions.
        # Multiply the matrices instead.
        self.assertTrue((mfrac.array * (~(warp.mask.array & bit_mask)) == 0).all())

    @lsst.utils.tests.methodParameters(mask_bitname=["BAD", "CR", "SAT"])
    def test_makeWarpMfrac(self, mask_bitname):
        """Test mfrac is being properly propagated"""

        config = MakeShearWarpConfig()
        nexpected = 4

        # this masks a single pixel, which then propagates to mfrac
        # in a set of neighboring pixels
        for exposureRef in self.dataRefs:
            fill_exposure(
                exposureRef.get(),
                rng=self.rng,
                mask_pixel=True,
                mask_bitname=mask_bitname,
            )

        makeWarp = MakeShearWarpTask(config=config)
        result = makeWarp.run(
            self.dataRefs,
            skyInfo=self.skyInfo,
            visit_summary=None,
        )

        mfrac = result.mfrac_warp
        warp = result.warp

        wbad = np.where(mfrac.array != 0)
        self.assertEqual(wbad[0].size, nexpected)

        iflag = afw_image.Mask.getPlaneBitMask("INTRP")
        wintrp = np.where(warp.mask.array & iflag != 0)
        self.assertEqual(wintrp[0].size, nexpected)

    def test_makeWarpCompare(self):
        """
        Test the results match what we get calling warp_exposures directly

        We need a version of this that persists to disk as well, to ensure the
        round trip preserves the data
        """
        from descwl_coadd import warp_exposures

        config = MakeShearWarpConfig()

        # this masks a single pixel, which then propagates to mfrac
        # in a set of neighboring pixels
        for exposureRef in self.dataRefs:
            fill_exposure(
                exposureRef.get(),
                rng=self.rng,
                mask_pixel=True,
                mask_bitname="CR",
            )

        # this warps all exposures onto a common image, so they must not
        # overlap.  Our _make_data creates the images from CCD bounding boxes

        makeWarp = MakeShearWarpTask(config=config)
        result = makeWarp.run(
            self.dataRefs,
            skyInfo=self.skyInfo,
            visit_summary=None,
        )

        coadd_bbox, coadd_wcs = self.skyInfo.bbox, self.skyInfo.wcs
        for exposureRef in self.dataRefs:
            exp = exposureRef.get()

            data_id = exposureRef.dataId
            seed = MakeShearWarpTask.get_seed_from_data_id(data_id)

            rng = np.random.RandomState(seed + config.seed_offset)
            warped_exposures = warp_exposures(
                exp,
                coadd_wcs,
                coadd_bbox,
                remove_poisson=config.remove_poisson,
                rng=rng,
                bad_mask_planes=config.bad_mask_planes,
                verify=False,
            )

            # these warps are same size as big image in result
            warp, noise_warp, mfrac_warp, _ = warped_exposures

            # get portion that was written into
            msk = ~(np.isnan(warp.image.array))

            np.testing.assert_array_equal(
                warp.image.array[msk],
                result.warp.image.array[msk],
            )
            np.testing.assert_array_equal(
                warp.mask.array[msk],
                result.warp.mask.array[msk],
            )
            np.testing.assert_array_equal(
                warp.variance.array[msk],
                result.warp.variance.array[msk],
            )

            np.testing.assert_array_equal(
                noise_warp.image.array[msk],
                result.noise0_warp.image.array[msk],
            )
            np.testing.assert_array_equal(
                noise_warp.mask.array[msk],
                result.noise0_warp.mask.array[msk],
            )
            np.testing.assert_array_equal(
                noise_warp.variance.array[msk],
                result.noise0_warp.variance.array[msk],
            )

            np.testing.assert_array_equal(
                mfrac_warp.image.array[msk],
                result.mfrac_warp.array[msk],
            )


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    """Check for resource leaks."""


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
