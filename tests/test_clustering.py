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
    nof_clusters = 10
    vectors, clusters = make_blobs(n_samples=1000, cluster_std=1,
                                   n_features=dimensions, centers=nof_clusters,
                                   shuffle=True)

    model = bfr.Model(dimensions=dimensions, nof_clusters=nof_clusters)
    model.create(vectors)

    def test_plot(self):
        """ predicts points of the generated testdata using the created bfr.model

        -------

        """
        predictions = numpy.zeros(1000)
        for idx, point in enumerate(self.vectors):
            predictions[idx] = self.model.predict(point)

        x_cord, y_cord = self.vectors.T
        #print(x)
        #colors = ['black', 'green', 'blue', 'orange', 'red', 'gray']
        #color_map = matplotlib.colors.ListedColormap(colors)
        #matplotlib.pyplot.scatter(x, y, c=predictions, cmap=color_map)
        matplotlib.pyplot.scatter(x_cord, y_cord, c=predictions)
        matplotlib.pyplot.show()


if __name__ == '__main__':
    unittest.main()
