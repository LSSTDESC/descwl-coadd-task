from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Iterable

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image as afw_image
import lsst.utils.tests
import numpy as np
from lsst.afw.detection import GaussianPsf
from lsst.cell_coadds import (
    CoaddUnits,
    CommonComponents,
    MultipleCellCoadd,
    PatchIdentifiers,
)
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.coaddBase import makeSkyInfo

from descwl_coadd_task import (
    AssembleShearCoaddSlowTask,
    MakeShearWarpConfig,
    MakeShearWarpTask,
)
from descwl_coadd_task.utils_for_tests import (
    build_skyMap,
    construct_geometry,
    fill_exposure,
    generate_data_id,
    generate_photoCalib,
)

if TYPE_CHECKING:
    from lsst.cell_coadds import ObservationIdentifiers


PIXEL_SCALE = 0.2
"""Pixel scale in arcseconds."""


class MockAssembleShearCoaddSlowTask(AssembleShearCoaddSlowTask):
    """Mock a PipelineTask bypassing the middleware for testing."""


class AssembleCoaddTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.skyMap = build_skyMap()
        cls.photoCalib = generate_photoCalib()
        cls.skyInfo = makeSkyInfo(cls.skyMap, tractId=3828, patchId=41)

        cls.dataRefsDict = {
            visit_id: construct_geometry(visit_id) for visit_id in (204706, 204708)
        }

        cls.common = CommonComponents(
            units=CoaddUnits.nJy,
            wcs=cls.skyInfo.patchInfo.wcs,
            band="fakeFilter",
            identifiers=PatchIdentifiers(
                skymap="cells", tract=3828, patch=41, band="fakeFilter"
            ),
        )

    def setUp(self):
        self.rng = np.random.RandomState(12345)

        # Fill the input exposures.
        # This needs to be redone for each test, since the test may have
        # modified the pixel values in-place.
        for visit, exposureRefs in self.dataRefsDict.items():
            for exposureRef in exposureRefs:
                fill_exposure(exposureRef.get(), self.rng)

        makeWarpConfig = MakeShearWarpConfig()
        makeWarp = MakeShearWarpTask(config=makeWarpConfig)

        self.warpDict, self.mfracWarpDict, self.noiseWarpDict = {}, {}, {}
        for visit in self.dataRefsDict:
            warp_result = makeWarp.run(
                self.dataRefsDict[visit], self.skyInfo, visit_summary=None
            )
            self.warpDict[visit] = InMemoryDatasetHandle(
                warp_result.warp, dataId=self.dataRefsDict[visit][0].dataId
            )
            self.mfracWarpDict[visit] = InMemoryDatasetHandle(
                warp_result.mfrac_warp, dataId=self.dataRefsDict[visit][0].dataId
            )

            # The variance plane of noise warps get modified in-place.
            # When run against an actual butler, this is not an issue since
            # it is read in from disk/object storage everytime. We mimic this
            # behavior by requiring that we get a new copy every time get() is
            # called.
            self.noiseWarpDict[visit] = InMemoryDatasetHandle(
                warp_result.noise0_warp,
                dataId=self.dataRefsDict[visit][0].dataId,
                copy=True,
            )

    def tearDown(self):
        for _, exposureRefs in self.dataRefsDict.items():
            for exposureRef in exposureRefs:
                exposureRef.get().mask.clearAllMaskPlanes()

    @classmethod
    def _constdfgdfgruct_geometry(cls, visit_id: int):
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
                    f"{visit_id} is not supported. It must be one of "
                    "[1248, 204706, 397330]"
                )

        dataRefs = []

        for detId, detectorNum in enumerate(detectorNums):
            wcs = lsst.afw.geom.makeSkyWcs(crpixList[detId], crvalList[detId], cdMatrix)
            exposure = afw_image.ExposureF(cls.nx, cls.ny)

            # Set the PhotoCalib, Wcs and detector objects of this exposure.
            exposure.setPhotoCalib(cls.photoCalib)
            exposure.setPsf(GaussianPsf(25, 25, 2.5))
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

            dataId_dict = {
                "detector_id": detectorNum,
                "visit_id": visit_id,
                "band": "fake",
            }
            dataId = generate_data_id(**dataId_dict)
            dataRef = InMemoryDatasetHandle(exposure, dataId=dataId)

            dataRefs.append(dataRef)

        return dataRefs

    def checkSortOrder(self, inputs: Iterable[ObservationIdentifiers]) -> None:
        """Check that the inputs are sorted.

        The inputs must be sorted first by visit, and within the same visit,
        by detector.

        Parameters
        ----------
        inputs : `Iterable` [`ObservationIdentifiers`]
            The inputs to be checked.
        """
        visit, detector = -np.inf, -np.inf  # Previous visit, detector IDs.
        for _, obsId in enumerate(inputs):
            with self.subTest(input_number=obsId):
                self.assertGreaterEqual(obsId.visit, visit)
            if visit == obsId.visit:
                with self.subTest(detector_number=obsId.detector):
                    self.assertGreaterEqual(obsId.detector, detector)

            visit, detector = obsId.visit, obsId.detector

    def checkPsfImage(self, psf_image):
        """Check that the PSF image is a valid image."""

        psf_dimensions = psf_image.getDimensions()
        self.assertEqual(psf_dimensions.x, self.task.config.psf_dimensions)
        self.assertEqual(psf_dimensions.y, self.task.config.psf_dimensions)

        array = psf_image.array
        self.assertGreaterEqual(array.min(), 0)  # Pixel values must be non-negative.
        self.assertLessEqual(array.sum(), 1)
        self.assertGreaterEqual(array.sum(), 0.995)  # It does not sum to 1 exactly.

    def checkRun(self, result):
        """Check that the task runs successfully."""

        max_visit_count = len(self.warpDict)

        # Check that we produced a MultipleCellCoadd instance.
        self.assertTrue(isinstance(result.multiple_cell_coadd, MultipleCellCoadd))
        for cellId, single_cell_coadd in result.multiple_cell_coadd.cells.items():
            # Use subTest context manager so this gets run for each cell
            # and collect all failures.
            with self.subTest(x=cellId.x, y=cellId.y):
                # Check that the visit_count method returns a number less than
                # or equal to the total number of input exposures available.
                self.assertLessEqual(single_cell_coadd.visit_count, max_visit_count)

                # Check that the inputs are sorted.
                self.checkSortOrder(single_cell_coadd.inputs)

                # Check that the PSF image
                self.checkPsfImage(single_cell_coadd.psf_image)

    def testCoaddSmoke(self):
        config = MockAssembleShearCoaddSlowTask.ConfigClass()
        self.task = MockAssembleShearCoaddSlowTask(config=config)
        # task is an attribute so the check methods can access its config.

        result = self.task.run(
            input_warps=self.warpDict.values(),
            noise_warps=self.noiseWarpDict.values(),
            mfrac_warps=self.mfracWarpDict.values(),
            skyInfo=self.skyInfo,
            common=self.common,
        )

        self.checkRun(result)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    """Check for resource leaks."""


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

    # Run the test cases when invoked as a script.
    tc = AssembleCoaddTestCase()
    tc.setUpClass()
    tc.setUp()
    tc.testCoaddSmoke()
