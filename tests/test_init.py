"""Testing template"""
import unittest
import numpy
from .context import bfr


class BasicTests(unittest.TestCase):
    """Basic test cases."""
    dimensions = 2
    nof_points = 5
    nof_clusters = 5
    model = bfr.Model(dimensions=dimensions,
                      nof_clusters=nof_clusters,
                      mahalanobis_threshold=3)
    def test_init_cluster(self):
        """ Passed if a cluster is initialised with the correct attributes

        -------

        """

        dimensions = 10
        cluster = bfr.Cluster(dimensions)
        self.assertEqual(cluster.size, 0, "Cluster size not initially 0")
        self.assertEqual(numpy.sum(cluster.sums), 0, "Cluster sums not initially 0 vector")
        self.assertEqual(numpy.sum(cluster.sums_sq), 0, "Cluster sums_sq not initially 0 vector")

    def test_init_model(self):
        """ Test that a bfr model gets initialized correctly

        -------

        """
        model = self.model
        self.assertEqual(model.dimensions, self.dimensions, "Incorrect dimensionality")
        self.assertEqual(model.discard, [], "Incorrect discard_set")
        self.assertEqual(model.compress, [], "Incorrect compress set")
        self.assertEqual(model.retain, [], "Incorrect retain set")

    def test_random_points(self):
        """ Tests that the appropriate amount of random points are returned and
        that that the remaining points matrix is adjusted appropriately.

        -------

        """
        points = numpy.ones((self.nof_points, self.dimensions))
        nof_clusters = 4
        for i in range(self.nof_points):
            points[i] *= i
        bfr.random_points(nof_clusters, points, 1)
        used = 0
        for point in points:
            if bfr.used(point):
                print(used)
                used += 1
        self.assertEqual(nof_clusters, 4, "Remaining points incorrect")

    def test_variance(self):
        """ Tests that the has_variance flag of a cluster updates accordingly.

        -------

        """
        cluster = bfr.Cluster(2)
        bfr.update_cluster(numpy.ones(2), cluster)
        self.assertFalse(cluster.has_variance, "Cluster has variance when it should not")
        bfr.update_cluster(numpy.ones(2) * 2, cluster)
        self.assertTrue(cluster.has_variance, "Cluster variance not updated")

if __name__ == '__main__':
    unittest.main()
