"""Tests for the module bfr.objective"""
import unittest
import numpy
from .context import bfr
from bfr import clustlib
from bfr import objective

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


class ObjectiveTests(unittest.TestCase):
    """Test cases for the module bfr.objective"""

    def test_finish_points(self):
        """ Tests that objective.finish_points evaluates to true when called with
        the last row index of points.

        -------

        """
        points = numpy.zeros((2, DIMENSIONS))
        points[0] = point
        points[1] = other_point
        done = objective.finish_points(len(points) - 1, points, '_')
        self.assertTrue(done, "Finish points")

    def test_zerofree_variances(self):
        """ Tests that objective.zerofree_variances evaluates to true when all clusters in
        model.discard has a non zero variance in each dimension


        -------

        """
        model.discard.append(ones)
        model.discard.append(twos)
        zerofree = objective.zerofree_variances('_', '_', model)
        self.assertFalse(zerofree, "clusters in model.discard do not have non zero variances")
        merged = clustlib.merge_clusters(ones, twos)
        model.discard[1] = merged
        zerofree = objective.zerofree_variances('_', '_', model)
        self.assertFalse(zerofree, "clusters in model.discard do not have non zero variances")
        model.discard[0] = merged
        zerofree = objective.zerofree_variances('_', '_', model)
        self.assertTrue(zerofree, "Clusters in model.discard has non zero variances")


if __name__ == '__main__':
    unittest.main()