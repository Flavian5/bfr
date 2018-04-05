"""Tests for the module bfr.setlib"""
import unittest
import numpy
from .context import bfr
from bfr import clustlib
from bfr import setlib


DIMENSIONS = 2
NOF_POINTS = 5
NOF_CLUSTERS = 5
model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=5.0,
                  merge_threshold=10.0, dimensions=DIMENSIONS,
                  init_rounds=10, nof_clusters=NOF_CLUSTERS)
model.initialized = True
INFINITY = 13371337.0
point = numpy.ones(2)
other_point = point * 2
ones = clustlib.Cluster(2)
twos = clustlib.Cluster(2)
clustlib.update_cluster(point, ones)
clustlib.update_cluster(other_point, twos)
arbitrary_set = [ones, ones]


class SetlibTests(unittest.TestCase):
    """Test cases for the bfr.setlib module"""

    def test_try_retain(self):
        """ Tests if a point is included in the retain set if distance to the closest
        retain set cluster is bigger than threshold.

        Tests if a point is merged with its closest retain set cluster if distance
        is less than threshold.

        Tests that two merged clusters get moved to the compress set.

        -------

        """
        model.retain = []
        model.retain.append(ones)
        model.eucl_threshold = -1
        setlib.try_retain(point, model)
        self.assertEqual(len(model.retain), 2, "Point incorrectly merged")
        model.eucl_threshold = INFINITY
        setlib.try_retain(point, model)
        self.assertLess(len(model.retain), 2, "Point incorrectly not merged")
        self.assertEqual(len(model.compress), 1, "Point not moved to compress")
        model.retain = []
        model.compress = []

    def test_try_include(self):
        """ Tests if try_include detects a used point and empty cluster_set.

        Tests that the point gets included when threshold = INFINITY.

        Tests that the point is disregarded when threshold = -1

        -------

        """
        used = numpy.ones(DIMENSIONS)
        used[0] = numpy.nan
        self.assertTrue(setlib.try_include(used, arbitrary_set, model), "Used point not detected")
        self.assertFalse(setlib.try_include(point, [], model), "Empty set not detected")
        model.threshold = -1
        self.assertFalse(setlib.try_include(point, arbitrary_set, model), "Incorrectly included")
        model.threshold = INFINITY
        self.assertTrue(setlib.try_include(point, arbitrary_set, model), "Incorrectly excluded")
        self.assertGreater(arbitrary_set[0].size, 1, "Set not updated")
        arbitrary_set[0] = clustlib.Cluster(DIMENSIONS)
        clustlib.update_cluster(point, arbitrary_set[0])
        model.threshold = model.eucl_threshold

    def test_finalize_set(self):
        """ Tests that an arbitrary set updates the cluster in model.discard with
        the closest center.

        -------

        """

        model.discard.append(ones)
        model.discard.append(twos)
        setlib.finalize_set(arbitrary_set, model)
        zeros = model.discard[0]
        self.assertEqual(zeros.size, 3, "Incorrect discard cluster updated")
        model.discard = []

    def test_update_compress(self):
        """ Tests that no clusters in the compress set gets merged when merge_threshold is negative.
        Tests that all clusters in the compress set gets merged when merge_threshold is huge.

        -------

        """
        has_std = clustlib.merge_clusters(ones, twos)
        model.compress.append(has_std)
        model.compress.append(has_std)
        model.merge_threshold = -1.0
        setlib.update_compress(model)
        nof_clusters = len(model.compress)
        self.assertGreater(nof_clusters, 1, "Compress set Clusters merged when they should not")

        model.merge_threshold = INFINITY
        setlib.update_compress(model)
        nof_clusters = len(model.compress)
        self.assertLess(nof_clusters, 2, "Compress set Clusters did not merge when they should")
        model.merge_threshold = INFINITY
        model.compress = []


if __name__ == '__main__':
    unittest.main()
