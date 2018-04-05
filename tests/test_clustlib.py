"""Tests for the module bfr.modellib"""
import unittest
import numpy
from .context import bfr
from bfr import clustlib
from bfr import ptlib
from functools import reduce
from sklearn.datasets.samples_generator import make_blobs



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
    """Test cases for the module bfr.clustlib"""

    def test_variance(self):
        """ Tests that the has_variance flag of a cluster updates accordingly.

        -------

        """
        cluster = clustlib.Cluster(2)
        clustlib.update_cluster(numpy.ones(2), cluster)
        self.assertFalse(cluster.has_variance, "Cluster has variance when it should not")
        clustlib.update_cluster(numpy.ones(2) * 2, cluster)
        self.assertTrue(cluster.has_variance, "Cluster variance not updated")

    def test_update_cluster(self):
        """ Tests that a cluster get its attributes updated accordingly.


        -------

        """
        merged = clustlib.merge_clusters(ones, ones)
        size_b4 = merged.size
        sums_b4 = merged.sums[0]
        sums_sq_b4 = merged.sums_sq[0]
        self.assertFalse(clustlib.has_variance(merged), "Cluster has no variance")
        for i in range(5):
            i += 1
            clustlib.update_cluster(other_point, merged)
            self.assertEqual(size_b4 + i, merged.size, "Incorrect size")
            self.assertEqual(sums_b4 + i * 2, merged.sums[0], "Incorrect sum")
            self.assertEqual(sums_sq_b4 + (i * 4), merged.sums_sq[0], "incorrect sums_sq")
            self.assertTrue(merged.has_variance, "Cluster has variance")

    def test_closest(self):
        """ Tests that the index of cluster with the closest mean is identified.


        -------

        """
        model.discard.append(ones)
        model.discard.append(twos)
        idx = clustlib.closest(point, model.discard, clustlib.euclidean)
        other = clustlib.closest(other_point, model.discard, clustlib.euclidean)
        self.assertEqual(idx, 0, "Incorrect closest index")
        self.assertEqual(other, 1, "Incorrect closest index")
        model.discard = []

    def test_merge_clusters(self):
        """ Tests that a merged cluster gets attributes according to the
         value of the merged clusters


        -------

        """
        merged = clustlib.merge_clusters(ones, twos)
        for i in range(5):
            size_b4 = merged.size + merged.size
            sums_b4 = merged.sums + merged.sums
            sums_sq_b4 = merged.sums_sq + merged.sums_sq
            has_var_b4 = merged.has_variance & merged.has_variance
            merged = clustlib.merge_clusters(merged, merged)
            self.assertEqual(size_b4, merged.size, "Incorrect size")
            self.assertEqual(sums_b4[0], merged.sums[0], "Incorrect sums")
            self.assertEqual(sums_sq_b4[1], merged.sums_sq[1], "Incorrect sums_sq")
            self.assertTrue(has_var_b4, "Incorrect sums")

    def test_has_variance(self):
        """ Tests that has_variance only evaluates to true when it has a non zero std_dev in
        each dimension


        -------

        """
        no_variance = clustlib.has_variance(ones)
        cluster = clustlib.Cluster(DIMENSIONS)
        other = clustlib.Cluster(DIMENSIONS)
        clustlib.update_cluster(point, cluster)
        clustlib.update_cluster(point, other)
        point_a = numpy.ones(2)
        point_b = numpy.ones(2)
        point_a[1] = 2
        point_b[0] = 2
        clustlib.update_cluster(point_a, cluster)
        clustlib.update_cluster(point_b, other)
        merged = clustlib.merge_clusters(other, cluster)
        self.assertFalse(no_variance, "No dimension has variance")
        self.assertFalse(clustlib.has_variance(cluster), "Second dimension has variance")
        self.assertFalse(clustlib.has_variance(other), "First dimension has variance")
        self.assertTrue(clustlib.has_variance(merged), "Both dimensions have variance")

    def test_mean(self):
        """ Tests if the mean of 1, 2 is 1.5 and that the mean of 1 is 1


        -------

        """
        merged = clustlib.merge_clusters(ones, twos)
        self.assertEqual(clustlib.mean(merged)[0], 1.5, "Incorrect mean")
        self.assertEqual(clustlib.mean(ones)[1], 1, "Incorrect mean")
    def test_euclidean(self):
        """ Tests that two different methods of computing Euclidean distance computes to
        the same value.


        -------

        """
        dist = clustlib.euclidean(point, twos)
        alternative = point - other_point
        alternative = alternative ** 2
        alternative = numpy.sum(alternative)
        alternative = numpy.sqrt(alternative)
        self.assertEqual(dist, alternative, "Different Euclidean distances")

    def test_sum_squared_diff(self):
        """ Tests that sum of squared diff of {1,1} and {2,2} is equal to
        (1 - 2) ^ 2 + (1 - 2) ^ 2 = 2.
        Tests that the sum of squared diff of {3,3} and {1,1} is equal to
        (3 - 1) ^ 2 + (3 - 1) ^ 2 = 8.


        -------

        """
        diff = clustlib.sum_squared_diff(point, twos)
        self.assertEqual(diff, 2, "Incorrect sum of squared diff")
        diff = clustlib.sum_squared_diff(point * 3, ones)
        self.assertEqual(diff, 8, "Incorrect second sum of squared diff")

    def test_mahalanobis(self):
        """ Tests that mahalanobis distance computes to the sum over dimensions of
        euclidean distances divided by the standard deviation


        -------

        """
        merged = clustlib.merge_clusters(ones, twos)
        dist = clustlib.mahalanobis(point, merged)
        alternative = point - clustlib.mean(merged)
        alternative /= clustlib.std_dev(merged)
        alternative = alternative ** 2
        alternative = numpy.sum(alternative)
        alternative = numpy.sqrt(alternative)
        self.assertEqual(dist, alternative, "Incorrect distance")

    def test_cluster_point(self):
        """ Tests that the closest cluster is updated and that a used point is detected.


        -------

        """
        another = clustlib.Cluster(2)
        clustlib.update_cluster(point, another)
        model.discard.append(another)
        model.discard.append(twos)
        clustlib.cluster_point(point, model)
        self.assertEqual(model.discard[0].size, 2)
        nan = numpy.zeros(2)
        nan[0] = numpy.nan
        clustlib.cluster_point(nan, model)
        model.discard = []

    def test_std_check(self):
        """ Tests that the std_check evaluates to true for a set of {{1,1}, {1,1}}
        Tests that the std_check evaluates to false when threshold is set to 0 and
        at least one of the checked clusters have a non zero variance


        -------

        """
        has_std = clustlib.merge_clusters(ones, twos)
        should_merge = clustlib.std_check(ones, ones, 1)
        should_not = clustlib.std_check(has_std, ones, 0)
        self.assertTrue(should_merge, "Incorrect False")
        self.assertFalse(should_not, "Incorrect True")

    def test_std_dev(self):
        """ Tests that two computed population standard deviations compute to the same value
        as when computing with numpy


        -------

        """
        yet_another = numpy.ones(2)
        yet_another = yet_another * 4
        fours = clustlib.Cluster(2)
        clustlib.update_cluster(yet_another, fours)
        merged = clustlib.merge_clusters(fours, ones)
        std_dev = clustlib.std_dev(merged)
        self.assertEqual(std_dev[0], numpy.std([1, 4]), "Incorrect std_dev")
        clustlib.update_cluster(point * 5, merged)
        std_dev = clustlib.std_dev(merged)
        res = numpy.std([1, 4, 5])
        std_dev[1] = round(std_dev[1], 5)
        res = round(res, 5)
        self.assertEqual(std_dev[1], res, "Incorrect std_dev")
