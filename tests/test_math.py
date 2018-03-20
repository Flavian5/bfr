"""Testing mathematical functions used in bfr"""
import unittest
import random
import numpy
from .context import bfr


class MathTests(unittest.TestCase):
    """Mathematical test cases"""
    dimensions = 3
    no_clusters = 3
    point = numpy.ones(dimensions)
    clusters = numpy.empty(no_clusters, dtype=bfr.Cluster)
    for i in range(no_clusters):
        clusters[i] = bfr.Cluster(dimensions)
        for j in range(dimensions):
            number = random.randint(-1337, 1337)
            clusters[i].sums[j] = number
            clusters[i].sums_sq[j] = numpy.square(number)
            clusters[i].size = 2
    cluster = clusters[0]

    def test_update_cluster(self):
        """ Tests that a cluster updates its metadata correctly
        sum[0] = 1 + 2 = 3
        sums_sq[0] = 1² + 2² = 5
        mean = 1 + 2 / 2 = 1.5
        std_dev = sqrt(5 / 2 - (3 / 2)²) = 0.5
        -------

        """
        cluster = bfr.Cluster(self.dimensions)
        point = self.point
        bfr.update_cluster(point, cluster)
        bfr.update_cluster(point * 2, cluster)
        self.assertEqual(cluster.sums[0], 3, "Incorrect sum")
        self.assertEqual(cluster.sums_sq[0], 5, "Incorrect sums_sq")
        self.assertEqual(bfr.mean(cluster)[0], 1.5, "Incorrect mean")
        self.assertEqual(bfr.std_dev(cluster)[0], 0.5, "Incorrect std_dev")

    def test_closest(self):
        """ Tests that no other centroid is closer than the suggested closest
        -------

        """
        closest_idx = bfr.closest(self.point, self.clusters, bfr.euclidean)
        closest_cluster = self.clusters[closest_idx]
        for cluster in self.clusters:
            min_dist = bfr.euclidean(self.point, closest_cluster)
            distance = bfr.euclidean(self.point, cluster)
            self.assertLessEqual(min_dist, distance, "Incorrect closest cluster")

    def test_std_dev(self):
        """ TODO update test case

        -------

        """
        print(bfr.std_dev(self.cluster))

    def test_mean(self):
        """ Passed if the mean vector is computed correctly
        -------

        """
        mean = self.cluster.sums / self.cluster.size
        diff = mean - bfr.mean(self.cluster)
        for dimension in diff:
            self.assertEqual(dimension, 0, "Mean vector incorrect")

    def test_euclidean(self):
        """ Passed if the euclidean distance is computed correctly

        -------

        """
        euclidean = bfr.euclidean(self.point, self.cluster)
        diff = self.point - bfr.mean(self.cluster)
        other_euclidean = numpy.linalg.norm(diff)
        self.assertEqual(euclidean, other_euclidean, "Incorrect euclidean distance")


    def test_mahalanobis(self):
        """ Passed if the mahalanobis distance computes to the same value
        with two alternative methods

        -------

        """
        # First computation
        total = 0
        mean = bfr.mean(self.cluster)
        std_dev = bfr.std_dev(self.cluster)
        diff = self.point - mean
        for i in range(self.dimensions):
            normalised = diff[i] / std_dev[i]
            squared = normalised ** 2
            total = total + squared
        result = numpy.sqrt(total)
        # Second computation
        mahalanobis = bfr.mahalanobis(self.point, self.cluster)
        self.assertEqual(result, mahalanobis, "Differing mahalanobis distances")


if __name__ == '__main__':
    unittest.main()
