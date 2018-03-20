"""This is a module defining bfr"""
import numpy
import bfr


class Model:
    """ A bfr model

    Attributes
    ----------
    mahal_threshold : float
        Nearness of point and cluster is determined by
        mahalanobis distance < threshold * sqrt(dimensions)

    eucl_threshold : float
        Nearness of two points is determined by Euclidean distance < threshold

    threshold : float
        The current default threshold used by the model

    merge_threshold : float
        Two sets will be merged if they

    threshold_fun : function
        The current default function for determining if a point and a cluster
        are considered close. The function should accept a point and a cluster
        and return a float.

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
        self.mahal_threshold = kwargs.pop('mahalanobis_factor', 3)
        self.eucl_threshold = kwargs.pop('euclidean_threshold', 0.2)
        self.merge_threshold = kwargs.pop('merge_threshold', 2.5)
        self.threshold = self.eucl_threshold
        self.threshold_fun = bfr.euclidean
        self.distance_fun = bfr.euclidean
        self.dimensions = kwargs.pop('dimensions')
        self.nof_clusters = kwargs.pop('nof_clusters')
        self.mahal_threshold = self.mahal_threshold * numpy.sqrt(self.dimensions)
        self.discard = []
        self.compress = []
        self.retain = []

    def initialize(self, points, initial_points=None):
        """ Initializes clusters using points and optionally specified initial points.

        Parameters
        ----------
        points : numpy.matrix
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        initial_points : numpy.matrix
            Matrix with rows that will be used as the initial centers.
            The points should have the same dimensionality as the model
            and the number of points should be equal to the number of clusters.

        Returns
        -------
        index : int
            Returns the row index to the first point not included in the model.
            Note : the point may be numpy.nan if it was randomly picked by
            random points.

        """

        if initial_points is None:
            initial_points = bfr.random_points(self.nof_clusters, points, 100)
        bfr.initiate_clusters(initial_points, self)
        return bfr.cluster_points(points, self, bfr.zerofree_variances)

    def update(self, points, idx=0):
        """

        Parameters
        ----------
        points : numpy.matrix
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        idx : int
            The first row index from which the model will be updated.

        Returns
        -------
        bool
            True if the clustering clusters all points. False otherwise

        """

        next_idx = bfr.cluster_points(points[idx:], self, bfr.finish_points)
        if next_idx:
            bfr.update_compress(self, 100)
            return True
        return False

    def create(self, points, initial_points=None):
        """ Creates a bfr model from points using the initial points specified in
        initial points.

        Parameters
        ----------
        points : numpy.matrix
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        initial_points : numpy.matrix
            Matrix with rows that will be used as the initial centers.
            The points should have the same dimensionality as the model
            and the number of points should be equal to the number of clusters.

        Returns
        -------
        bool
            True if the model was succesfully created. False otherwise.

        """

        if self.initialize(points, initial_points):
            self.threshold_fun = bfr.mahalanobis
            self.threshold = self.mahal_threshold
            self.update(points)
            return True
        return False

    def predict(self, points, outlier_detecion=False):
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
        nof_predictions = len(points)
        predictions = numpy.zeros(nof_predictions)

        for idx in range(nof_predictions):
            point = points[idx]
            predictions[idx] = bfr.predict_point(point, self, outlier_detecion)
        return predictions

    def finalize(self):
        bfr.finalize_set(self.compress, self)
        bfr.finalize_set(self.discard, self)

    def error(self, points):
        return bfr.eucl_error(self, points)
