"""Testing template"""
import unittest
from .context import bfr


class BasicTests(unittest.TestCase):
    """Basic test cases."""

    def test_init_cluster(self):
        """
        Tests that initialized clusters has correct dimensionality and number of attributes
        """
        dimensions = 10
        cluster = bfr.init_cluster(dimensions)
        self.assertEqual(len(cluster), 2, "Cluster has incorrect number of attributes")
        self.assertEqual(len(cluster.mean), dimensions, "Mean has incorrect dimensionality")
        self.assertEqual(len(cluster.std_dev), dimensions, "Std_dev has incorrect dimensionality")


if __name__ == '__main__':
    unittest.main()
