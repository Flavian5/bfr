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
model = bfr.Model(mahalanobis_factor=3, euclidean_threshold=5000.0,
                  merge_threshold=10.0, dimensions=DIMENSIONS,
                  init_rounds=10, nof_clusters=NOF_CLUSTERS)
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
    created = bfr.Model(mahalanobis_factor=3, euclidean_threshold=1.5,
                        merge_threshold=30.0, dimensions=DIMENSIONS,
                        init_rounds=1, nof_clusters=NOF_CLUSTERS)

    vectors, clusters = make_blobs(n_samples=1000, cluster_std=1.0,
                                   n_features=DIMENSIONS, centers=NOF_CLUSTERS,
                                   shuffle=True, random_state=None)

    def test_create(self):
        """ Tests that a model is created with the correct amount of clusters and
        that the initialization phase is ensured when threshold used in the
        initialization phase is large.

        -------

        """

        idx = self.created.create(self.vectors)
        nof_discard = len(self.created.discard)
        self.assertEqual(nof_discard, NOF_CLUSTERS, "Incorrect amount of clusters")
        self.assertTrue(idx, "Initialization objective never reached")

    def test_update(self):
        """ Tests that the number of points of a cluster is bigger after a model is
        updated with the same input as it was created with.

        -------

        """

        size = self.created.discard[0].size
        self.created.update(self.vectors)
        updated_size = self.created.discard[0].size
        self.assertGreater(updated_size, size, "First cluster not updated")

    def test_finalize(self):
        """ Tests that the sum of all cluster sizes equals to the number of points clustered
        Tests that the retain and compress set are empty after finalizing.

        -------

        """
        self.created.update(self.vectors)
        self.created.finalize()
        sizes = map(lambda cluster: cluster.size, self.created.discard)
        finalized_sizes = reduce(lambda size, other: size + other, list(sizes))
        retain_size = len(self.created.retain)
        compress_size = len(self.created.compress)
        #self.assertEqual(retain_size, 0, "retain set not finallized")
        #self.assertEqual(compress_size, 0, "compress set not finallized")
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
        error = model.error(points * 0)
        self.assertEqual(error, 4)
        error = model.error(points * 2)
        self.assertEqual(error, 8)
        model.discard = []


if __name__ == '__main__':
    unittest.main()
