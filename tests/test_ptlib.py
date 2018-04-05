"""Tests for the module bfr.ptlib"""
import unittest
import numpy
from .context import bfr
from bfr import clustlib
from bfr import ptlib


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


class PtlibTests(unittest.TestCase):
    """Test cases for the module bfr.ptlib"""


class PtlibTests(unittest.TestCase):

    def test_squared_diff(self):
        """ Tests that the sum of dimensions of ({1, 1} - {2, 2}) ^ 2 == 2


        -------

        """
        sum_sq_diff = ptlib.sum_squared_diff(point, other_point)
        self.assertEqual(sum_sq_diff, 2)
    def test_euclidean(self):
        """ Tests that the distance of {1,1} and {2,2} is sqrt(2)


        -------

        """
        dist = ptlib.euclidean(point, other_point)
        self.assertEqual(dist, numpy.sqrt(2), "Incorrect distance computed")

    def test_sum_all_euclideans(self):
        """ Tests that the dist(x1, x2) + dist(x2, x1) when x1={0, 0} and x2={0, 0} is 0
        Tests that the dist(x1, x2) + dist(x2, x1) when x1={1, 0} and x2={0, 0} is 2

        -------

        """
        points = numpy.ones((2, 2))
        summed = ptlib.sum_all_euclideans(points)
        self.assertEqual(summed, 0, "Incorrectly found zero distance")
        points[0][0] = 2
        summed = ptlib.sum_all_euclideans(points)
        self.assertEqual(summed, 2, "Incorrectly computed 2* eucl_dist = 1 + 1")

    def test_used(self):
        """ Tests that a point is marked and detected as used.


        -------

        """
        points = numpy.ones((DIMENSIONS, 3))
        points[0] = numpy.nan
        self.assertTrue(ptlib.used(points[0]), "Marking as used not working")

    def test_random_points(self):
        """ Tests that the appropriate amount of random points are returned and
        that that the remaining points matrix is adjusted appropriately.

        -------

        """
        points = numpy.ones((NOF_POINTS + 1, DIMENSIONS))
        for i in range(NOF_POINTS):
            points[i] *= i
        ptlib.random_points(points, model, 1)
        used = 0
        for point in points:
            if ptlib.used(point):
                used += 1
        self.assertEqual(NOF_CLUSTERS, used, "Remaining points incorrect")


    def test_best_spread(self):
        model.init_rounds = 1
        model.nof_clusters = 5
        points = numpy.ones((2, 2))
        points[0] = point
        points[1] = point

        result = ptlib.best_spread(points, model)
        self.assertTrue(isinstance(result, numpy.ndarray))


    def test_max_mindist(self):
        """ Tests that the index {6,6} is the furthest vector from {1,1} and {2,2}


        -------

        """

        points = numpy.ones((6, 2))
        points[0] = point
        points[1] = point * 2
        points[2] = point * 3
        points[3] = point * 4
        points[4] = point * 5
        points[5] = point * 6
        idx = ptlib.max_mindist(points, [0, 1], [2, 3, 4, 5])
        self.assertEqual(idx, 5, "Incorrect furthest index")


if __name__ == '__main__':
    unittest.main()
