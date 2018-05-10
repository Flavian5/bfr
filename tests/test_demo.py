"""This module tests clustering"""
import unittest
import matplotlib.pyplot
from sklearn.datasets.samples_generator import make_blobs
from .context import bfr


class TestClustering(unittest.TestCase):
    """Testing the outcome of clustering"""
    #Generating test data
    dimensions = 2
    nof_clusters = 5
    vectors, clusters = make_blobs(n_samples=10000, cluster_std=0.1,
                                   n_features=dimensions, centers=nof_clusters,
                                   shuffle=True)

    model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=0.3,
                      merge_threshold=0.01, dimensions=dimensions,
                      init_rounds=100, nof_clusters=5)
    model.fit(vectors[:5000])
    model.fit(vectors[5000:])
    model.finalize()
    print(model)
    print(model.error(vectors))
    print(model.error())

    def test_plot(self):
        """ predicts points of the generated testdata using the created bfr.model
        -------
        """

        self.model.plot(points=self.vectors, outlier_detection=True)


if __name__ == '__main__':
    unittest.main()
