"""Testing template"""
import unittest
import numpy
from .context import bfr


class BasicTests(unittest.TestCase):
    """Basic test cases."""
    dimensions = 2
    nof_points = 10
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
        points = numpy.ones((self.nof_points, self.dimensions))
        for i in range(self.nof_points):
            points[i] *= i
        initial_points = bfr.random_points(5, points)
        idx = points
        used = 0
        for point in points:
            if bfr.used(point):
                used += 1
        self.assertEqual(used, self.nof_points - self.nof_clusters, "Remaining points incorrect")

    """def test_initialize(self):
        points = numpy.ones((self.nof_points, self.dimensions))
        for i in range(len(points)):
            points[i] *= i
        self.model.initialize(points)"""

    """def test_create_model(self):
        points = numpy.ones((self.nof_points, self.dimensions))
        model = bfr.Model(dimensions=2, nof_clusters=2)
        model.create(points)
        centers = model.discard
        for center in centers:
            print("means", bfr.mean(center))"""

    def test_variance(self):
        cluster = bfr.Cluster(2)
        bfr.update_cluster(numpy.ones(2), cluster)
        self.assertFalse(cluster.has_variance, "Cluster har variance when it should not")
        bfr.update_cluster(numpy.ones(2) * 2, cluster)
        self.assertTrue(cluster.has_variance, "Cluster variance not updated")

if __name__ == '__main__':
    unittest.main()
