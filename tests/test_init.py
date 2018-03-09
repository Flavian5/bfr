"""Testing template"""
import unittest
import numpy
from .context import bfr


class BasicTests(unittest.TestCase):
    """Basic test cases."""

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
        model = bfr.Model(dimensions=3,
                          nof_clusters=4,
                          mahalanobis_threshold=3)
        self.assertEqual(model.dimensions, 3, "Incorrect dimensionality")
        self.assertEqual(model.mahal_threshold, 3, "Incorrect mahalanobis threshold")
        self.assertEqual(model.eucl_threshold, 1000, "Incorrect euclidean threshold")
        self.assertEqual(len(model.discard), model.nof_clusters, "Incorrect nof_clusters")
        self.assertEqual(model.compress, [], "Incorrect compress set")
        self.assertEqual(model.retain, [], "Incorrect retain set")

if __name__ == '__main__':
    unittest.main()
