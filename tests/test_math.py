"""Testing mathematical functions used in bfr"""
import unittest
import numpy
import random
from .context import bfr


class MathTests(unittest.TestCase):
    """Mathematical test cases"""
    dimensions = 3
    point = numpy.ones(dimensions)
    cluster =bfr.Cluster(dimensions)
    for i in range(dimensions):
        number = i + 1
        print(number)
        cluster.sums[i] = number
        cluster.sums_sq[i] = numpy.square(number)
    cluster.size = 2

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

    def test_malahanobis(self):
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
        mahalanobis = bfr.malahanobis(self.point, self.cluster)
        print(mahalanobis)
        self.assertEqual(result, mahalanobis, "Differing mahalanobis distances")


if __name__ == '__main__':
    unittest.main()
