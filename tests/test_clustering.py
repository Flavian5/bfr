"""This module tests clustering"""
import unittest
import numpy
import matplotlib.pyplot
from sklearn.datasets.samples_generator import make_blobs
from .context import bfr


class TestClustering(unittest.TestCase):
    """Testing the outcome of clustering"""
    #Generating test data
    dimensions = 2
    nof_clusters = 5
    vectors, clusters = make_blobs(n_samples=1000, cluster_std=1,
                                   n_features=dimensions, centers=nof_clusters,
                                   shuffle=True)

    model = bfr.Model(dimensions=dimensions, nof_clusters=nof_clusters)
    model.create(vectors)
    model.finalize()
    print(model.error(vectors))
    def test_plot(self):
        """ predicts points of the generated testdata using the created bfr.model

        -------

        """
        predictions = self.model.predict(self.vectors)
        x_cord, y_cord = self.vectors.T
        #print(x)
        #colors = ['black', 'green', 'blue', 'orange', 'red', 'gray']
        #color_map = matplotlib.colors.ListedColormap(colors)
        #matplotlib.pyplot.scatter(x, y, c=predictions, cmap=color_map)
        matplotlib.pyplot.scatter(x_cord, y_cord, c=predictions)
        matplotlib.pyplot.show()

    def test_merge(self):
        """ Passed if two clusters with equal sums and sums_sq has the same
        mean and std_dev as the merged clusters.
        sums ans sums_sq and


        -------

        """
        dims = 2
        cluster = bfr.Cluster(dims)
        point = numpy.ones(dims)
        # (1,1)
        bfr.update_cluster(point, cluster)
        other_point = point * 2
        # (2,2)
        bfr.update_cluster(other_point, cluster)

        mean = bfr.mean(cluster)
        std_dev = bfr.std_dev(cluster)
        sums = cluster.sums
        sums_sq = cluster.sums_sq
        has_variance = cluster.has_variance

        other_cluster = bfr.Cluster(dims)
        bfr.update_cluster(point, other_cluster)
        bfr.update_cluster(other_point, other_cluster)

        merged = bfr.merge_clusters(cluster, other_cluster)
        self.assertEqual(merged.has_variance, has_variance, "Incorrect has_variance flag after merge")
        merged_mean = bfr.mean(cluster)
        merged_std_dev = bfr.std_dev(cluster)
        for i in range(dims):
            self.assertEqual(merged_mean[i], mean[i], "Incorrect mean vector after merge")
            self.assertEqual(merged_std_dev[i], std_dev[i], "Incorrect std_dev vector after merge")

if __name__ == '__main__':
    unittest.main()
