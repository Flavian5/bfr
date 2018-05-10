"""Tests for the module bfr.modellib"""
import unittest
import numpy
from .context import bfr
from bfr import clustlib
from functools import reduce
from sklearn.datasets.samples_generator import make_blobs

DIMENSIONS = 2
NOF_POINTS = 5
NOF_CLUSTERS = 5
model = bfr.Model(mahalanobis_factor=3, euclidean_threshold=1.5,
                  merge_threshold=1.95, dimensions=DIMENSIONS,
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
    """Test cases for the module bfr.model"""
    created = bfr.Model(mahalanobis_factor=3, euclidean_threshold=3.5,
                        merge_threshold=30.0, dimensions=DIMENSIONS,
                        init_rounds=100, nof_clusters=NOF_CLUSTERS)

    vectors, clusters = make_blobs(n_samples=1000, cluster_std=1.0,
                                   n_features=DIMENSIONS, centers=NOF_CLUSTERS,
                                   shuffle=True, random_state=None)

    def test_fit(self):
        """ Tests that initalization phase is not completed when eucl_threshold is
        infinitely small. Tests that initialization phase is completed when eucl_threshold
        is infinite.


        -------

        """

        local = bfr.Model(mahalanobis_factor=3, euclidean_threshold=3.5,
                          merge_threshold=30.0, dimensions=DIMENSIONS,
                          init_rounds=5, nof_clusters=NOF_CLUSTERS)
        inf_small = 0.0001
        local.eucl_threshold = inf_small
        local.threshold = local.eucl_threshold
        local.distance_fun = clustlib.euclidean

        local.fit(self.vectors)
        self.assertTrue(local.threshold == inf_small, "Incorrectly switched threshold")
        local.eucl_threshold = INFINITY
        local.threshold = INFINITY
        local.fit(self.vectors)
        self.assertTrue(local.threshold != INFINITY, "Did not switch threshold")
        self.assertEqual(len(local.discard), NOF_CLUSTERS, "incorrect nr of clusters")

    def test_finalize(self):
        """ Tests that the sum of all cluster sizes equals to the number of points clustered
        Tests that the retain and compress set are empty after finalizing.

        -------

        """

        local = bfr.Model(mahalanobis_factor=3, euclidean_threshold=3.5,
                          merge_threshold=1.2, dimensions=DIMENSIONS,
                          init_rounds=5, nof_clusters=NOF_CLUSTERS)

        local.fit(self.vectors)
        local.fit(self.vectors)
        local.finalize()
        sizes = map(lambda cluster: cluster.size, local.discard)
        finalized_sizes = reduce(lambda size, other: size + other, list(sizes))
        retain_size = len(local.retain)
        compress_size = len(local.compress)
        self.assertEqual(retain_size, 0, "retain set not finallized")
        self.assertEqual(compress_size, 0, "compress set not finallized")
        self.assertEqual(finalized_sizes, 2000)

    def test_predict(self):
        """ Tests that predict identifies closest cluster successfully and that
        outlier detection works.

        -------

        """

        model.discard.append(ones)
        model.discard.append(twos)
        predictions = model.predict(points, False)
        self.assertEqual(predictions[0], 0, "Incorrect first prediction")
        self.assertEqual(predictions[1], 1, "Incorrect second prediction")
        model.threshold = -1
        predictions = model.predict(points, True)
        self.assertEqual(predictions[0], -1, "First outlier not detected")
        self.assertEqual(predictions[1], -1, "second outlier not detected")
        model.discard = []
        model.threshold = model.eucl_threshold

    def test_error(self):
        """ Tests that model.error correctly identifies closest cluster and
        computes the error correctly


        -------

        """

        model.discard.append(ones)
        model.discard.append(twos)
        rss_error = model.error(points * 0)
        std_error = model.error()
        self.assertTrue(isinstance(std_error, float))
        self.assertEqual(rss_error, 4)
        rss_error = model.error(points * 2)
        self.assertEqual(rss_error, 8)
        model.discard = []


if __name__ == '__main__':
    unittest.main()
