"""Tests for the module bfr.objective"""
import math
import unittest
import numpy
import scipy.stats
from .context import bfr
from bfr import clustlib
from bfr import objective
from bfr import plot

DIMENSIONS = 2
NOF_POINTS = 5
NOF_CLUSTERS = 5
model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=5.0,
                  merge_threshold=10.0, dimensions=DIMENSIONS,
                  init_rounds=10, nof_clusters=NOF_CLUSTERS)
model.initialized = True
INFINITY = 13371337.0
point = numpy.ones(2)
other_point = point * 2
ones = clustlib.Cluster(2)
twos = clustlib.Cluster(2)
clustlib.update_cluster(point, ones)
clustlib.update_cluster(other_point, twos)
arbitrary_set = [ones, ones]


class ObjectiveTests(unittest.TestCase):
    """Test cases for the module bfr.objective"""

    def test_show(self):
        """ Tests that a plot is shown


        -------

        """
        merged = clustlib.merge_clusters(ones, twos)
        model.discard.append(ones)
        model.discard.append(merged)
        bfr_plot = plot.BfrPlot(model)
        bfr_plot.show()

    def test_create_axis(self):
        """ Tests that axes are created with different projections


        -------

        """

        axis = plot.create_axis()
        axis_3d = plot.create_axis("3d")
        self.assertNotEqual(type(axis), type(axis_3d))

    def test_add_legend_entry(self):
        """ Tests that a legend entry is appended to legend entries.

        -------

        """

        legend_entries = []
        col = "blue"
        plot.add_legend_entry(legend_entries, col)
        self.assertEqual(len(legend_entries), 1, "No legend entry added")

    def test_find_points(self):
        """ Tests that the correct corresponding point is found.

        -------

        """
        predictions = numpy.ones(3)
        predictions[1] = 2
        points = numpy.zeros(3)
        points[1] = 1337
        corr_points = plot.find_points(points, predictions, 2)
        self.assertTrue(corr_points == 1337, "Incorrect corresponding point")

    def test_get_cluster_shape(self):
        """ Tests that a circular shape has the same maximum x and y value when the cluster is
        symmetric.

        -------

        """
        cluster = clustlib.merge_clusters(ones, twos)
        shape = plot.get_cluster_shape(model, cluster)
        max_x = numpy.max(shape[0])
        max_y = numpy.max(shape[1])
        max_x = round(max_x, 2)
        max_y = round(max_y, 2)
        self.assertEqual(max_x, max_y, "Incorrect position or dimension of circle")

    def test_confidence_interval(self):
        """Tests that a confidence interval is computed correctly.

        -------

        """
        cluster = clustlib.merge_clusters(ones, twos)
        threshold = 3 * numpy.sqrt(2)

        width, height = plot.confidence_interval(cluster, threshold)
        std_dev = clustlib.std_dev(cluster)
        conf_interval = scipy.stats.norm.interval(0.9975, scale=std_dev[0] / numpy.sqrt(2))
        delta = conf_interval[1] - conf_interval[0] - width
        delta = round(delta, 1)
        self.assertEqual(delta, 0.0, "Incorrect confidence interval")
if __name__ == '__main__':
    unittest.main()