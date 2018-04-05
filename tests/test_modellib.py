"""Tests for the module bfr.modellib"""
import unittest
import numpy
from .context import bfr
from bfr import clustlib
from bfr import objective
from bfr import modellib

DIMENSIONS = 2
NOF_POINTS = 5
NOF_CLUSTERS = 5
model = bfr.Model(mahalanobis_factor=3, euclidean_threshold=5000.0,
                  merge_threshold=10.0, dimensions=DIMENSIONS,
                  init_rounds=10, nof_clusters=NOF_CLUSTERS)
model.initialized = True
INFINITY = 13371337.0
point = numpy.ones(2)
other_point = point * 2
points = numpy.zeros((2, 2))
points[0] = point
points[1] = other_point
ones = clustlib.Cluster(2)
twos = clustlib.Cluster(2)
clustlib.update_cluster(point, ones)
clustlib.update_cluster(other_point, twos)
arbitrary_set = [ones, ones]


class ModellibTests(unittest.TestCase):
    """Test cases for the module bfr.modellib"""

    def test_initialize(self):
        """ Tests that a module is succesfully initialized and that objective is not reached
        when it should not be


        -------

        """
        idx = modellib.initialize(points, model, points)
        self.assertFalse(idx, "Incorrectly reached objectives")
        model.discard = []

    def test_initiate_clusters(self):
        """ Tests that the correct number of clusters are initiated and that their means
        are as expected.

        -------

        """
        modellib.initiate_clusters(points, model)
        self.assertEqual(len(model.discard), 2, "Incorrect number of initial clusters")
        mean = clustlib.mean(model.discard[0])
        other = clustlib.mean(model.discard[1])
        self.assertEqual(mean[0], 1, "Incorrect mean of first cluster")
        self.assertEqual(other[0], 2, "Incorrect mean of second cluster")
        model.discard = []

    def test_cluster_points(self):
        """ Tests that points get clustered correctly.
        Tests that the objective function works as intended.


        -------

        """

        model.discard.append(ones)
        model.discard.append(twos)
        done = modellib.cluster_points(points, model, objective.zerofree_variances)
        size = model.discard[0].size
        other_size = model.discard[1].size
        self.assertEqual(size, 2, "first cluster has wrong size")
        self.assertEqual(other_size, 2, "second cluster has wrong size")
        self.assertFalse(done, "Incorrectly reached objective")
        done = modellib.cluster_points(points, model, objective.finish_points)
        self.assertTrue(done, "Incorrectly did not reach objective")
        model.discard = []

    def test_predict_point(self):
        """ Tests that a zero vector identifies a cluster with the centroid (1,1) as closest.
        Tests that a point is identified as outlier if distance to closest cluster > threshold.


        -------

        """
        model.threshold = -1
        model.discard.append(ones)
        model.discard.append(twos)
        points = numpy.zeros((1, 2))
        self.assertEqual(model.predict(points, False)[0], 0, "incorrect prediction")
        self.assertEqual(model.predict(points, True)[0], -1, "Outlier not detected")
        model.threshold = model.eucl_threshold
        model.discard = []

    def test_rss_error(self):
        """ Tests that sum_sq of a model with two clusters with the means
        {1,1} and {2,2} in the discard set and two vectors {0,0} and {3,3} evaluates to
        (1 - 0) ^ 2 + (1 - 0) ^2 + (3 - 2) ^ 2 + (3 - 2) ^ 2 = 4
        Tests that points outside of threshold are successfully identified as outliers.

        -------

        """
        points = numpy.zeros((2, 2))
        points[0] = point * 0
        points[1] = point * 3
        model.discard.append(ones)
        model.discard.append(twos)

        error = modellib.rss_error(points, model, False)
        self.assertEqual(error, 4, "Incorrect sum of squares")
        model.threshold = numpy.sqrt(2)
        error = modellib.rss_error(points, model, True)
        self.assertEqual(error, 0, "Outlier detection incorrect")
        model.threshold = model.eucl_threshold
        model.discard = []


if __name__ == '__main__':
    unittest.main()