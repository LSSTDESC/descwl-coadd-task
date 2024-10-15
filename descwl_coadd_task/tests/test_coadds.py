from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Iterable

import lsst.afw.cameraGeom.testUtils
import lsst.utils.tests
import numpy as np
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
