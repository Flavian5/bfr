"""Testing mathematical functions used in bfr"""
import unittest
import numpy
import random
from .context import bfr


class MathTests(unittest.TestCase):
    """Mathematical test cases"""
    dimensions = 3
    point = numpy.ones(dimensions)
    cluster = bfr.init_cluster(dimensions)
    for i in range(dimensions):
        cluster.mean[i] = random.randint(0, 1337)
        cluster.std_dev[i] = random.randint(0, 1337)

    def test_malahanobis(self):
        """ Passed if the mahalanobis distance computes to the same value
        with two alternative methods

        -------

        """
        # First computation
        total = 0
        diff = self.point - self.cluster.mean
        for i in range(self.dimensions):
            normalised = diff[i] / self.cluster.std_dev[i]
            squared = normalised ** 2
            total = total + squared
        result = numpy.sqrt(total)
        # Second computation
        mahalanobis = bfr.malahanobis(self.point, self.cluster)
        self.assertEqual(result, mahalanobis, "Differing mahalanobis distances")


if __name__ == '__main__':
    unittest.main()
