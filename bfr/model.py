"""This is a module defining bfr"""
import sys
import numpy

from . import modellib
from . import objective
from . import setlib
from . import clustlib
from . import error

from functools import reduce

class Model:
    """ A bfr model

    Attributes
    ----------
    mahal_threshold : float
        Nearness of point and cluster is determined by
        mahalanobis distance < mahal_threshold * sqrt(dimensions)

    eucl_threshold : float
        Nearness of a point and a cluster is determined by
        Euclidean distance(point,cluster) < eucl_threshold

    threshold_fun : function
        The current default function for determining if a point and a cluster
        are considered close. The function should accept a point and a cluster
        and return a float.

    threshold : float
        The current default threshold used by the model. Should point to eucl_threshold when the
        current threshold_fun is Euclidean. Should point to mahal_threshold when the threshold_fun
        is mahalanobis.

    merge_threshold : float
        Two cluster in the compress set will be merged if their combined std_dev
        is less than merge_threshold in each dimension

    distance_fun : function
        The current default function for finding the closest cluster of a point.
        The function should accept a point and a cluster and return a float

    dimensions : int
        The dimensionality of the model

    nof_clusters : int
        The number of clusters (eg. K)

    discard : list
        The discard set holds all the clusters. A point will update a cluster
        (and thus be discarded) if it is considered near the cluster.

    compress : list
        The compression set holds clusters of points which are near to each other
        but not near enough to be included in a cluster of the discard set

    retain : list
        Contains uncompressed outliers which are neither near to other points nor a cluster

    """
    def __init__(self, **kwargs):
        self.mahal_threshold = kwargs.pop('mahalanobis_factor', 3.0)
        self.eucl_threshold = kwargs.pop('euclidean_threshold', "error")
        self.merge_threshold = kwargs.pop('merge_threshold', "error")
        self.init_rounds = kwargs.pop('init_rounds', 5)
        self.dimensions = kwargs.pop('dimensions', "error")
        self.nof_clusters = kwargs.pop('nof_clusters', "error")

        self.threshold_fun = clustlib.euclidean
        self.threshold = self.eucl_threshold
        self.distance_fun = clustlib.euclidean
        self.mahal_threshold = self.mahal_threshold * numpy.sqrt(self.dimensions)

        self.discard = []
        self.compress = []
        self.retain = []
        if not error.confirm_attributes(self):
            raise AttributeError("bfr.model has incorrect attributes")

    def create(self, input_points, initial_points=None):
        """ Creates a bfr model from input_points optionally using
        the initial points specified in initial points.
        Parameters
        ----------
        input_points : numpy.ndarray
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        initial_points : numpy.ndarray
            Matrix with rows that will be used as the initial centers.
            The points should have the same dimensionality as the model
            and the number of points should be equal to the number of clusters.

        Returns
        -------
        bool
            True if the model was succesfully created. False otherwise.

        """
        if not error.confirm_create(input_points, self):
            raise ValueError("Incorrect points or model attributes")
        points = numpy.copy(input_points)
        next_idx = modellib.initialize(points, self, initial_points)
        if next_idx:
            self.threshold_fun = clustlib.mahalanobis
            self.threshold = self.mahal_threshold
            self.update(points, next_idx)
            return True
        sys.stderr.write("Warning: bfr.Model created but never finished initialization phase")
        return False

    def update(self, input_points, next_idx=0):
        """ Updates a model given input_points. next_idx specifies which row of the input_points
        matrix to start with. Leave at 0 to include all points.

        Parameters
        ----------
        input_points : numpy.matrix
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        next_idx : int
            The next row index of points from which the model will be updated.

        Returns
        -------
        bool
            True if the clustering clusters all points. False otherwise

        """
        if not error.confirm_update(input_points, self):
            return False
        next_idx = modellib.cluster_points(input_points[next_idx:], self, objective.finish_points)
        if next_idx:
            setlib.update_compress(self)
            return True
        sys.stderr.write("Warning: All points not clustered")
        return False

    def finalize(self):
        if not error.confirm_attributes(self):
            sys.stderr.write("Warning: bfr.Model attributes tampered with")
            return
        setlib.finalize_set(self.compress, self)
        setlib.finalize_set(self.retain, self)
        self.compress = []
        self.retain = []


    def predict(self, points, outlier_detection=False):
        """ Predicts which cluster a point belongs to.

        Parameters
        ----------
        points : numpy.matrix
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        outlier_detection : bool
            If True, outliers will be predicted with -1.
            If False, predictions will not consider thresholds

        Returns
        -------
        idx : int
            The index of the closest cluster (defined by default distance_fun).
            Returns -1 if the point is considered an outlier (determined by
            default threshold_fun and threshold)

        """
        error.confirm_predict(points, self)
        nof_predictions = len(points)
        predictions = numpy.zeros(nof_predictions)
        for idx in range(nof_predictions):
            point = points[idx]
            predictions[idx] = modellib.predict_point(point, self, outlier_detection)
        return predictions.astype(int)

    def error(self, points):
        error.confirm_error(points, self)
        return modellib.rss_error(points, self)
