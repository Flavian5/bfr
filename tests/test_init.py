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


if __name__ == '__main__':
    unittest.main()
