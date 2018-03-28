""" This module contains functions mainly operating on/with bfr.models."""

from . import ptlib
from . import clustlib
from . import objective


def initialize(points, model, initial_points=None):
    """ Initializes clusters using points and optionally specified initial points.

    Parameters
    ----------
    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.numpy.matrix

    model : bfr.Model

    initial_points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    Returns
    -------
    next_idx : int
        Returns the row index to the first point not included in the model.
        Note : the point may be numpy.nan if it was randomly picked by
        random points.

    """

    if initial_points is None:
        initial_points = ptlib.best_spread(points, model, initial_points)
    initiate_clusters(initial_points, model)
    next_idx = cluster_points(points, model, objective.zerofree_variances)
    return next_idx


def initiate_clusters(initial_points, model):
    """ Updates the model with the initial cluster centers specified in initial_points.

    Parameters
    ----------
    initial_points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    model : bfr.Model

    """

    for point in initial_points:
        cluster = clustlib.Cluster(model.dimensions)
        clustlib.update_cluster(point, cluster)
        model.discard.append(cluster)


def cluster_points(points, model, objective_fun):
    """ Update a model with the given points. Finished when objective is reached.

    Parameters
    ----------
    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    model : bfr.Model

    objective_fun : function
        The objective_fun determines when a clustering has been succesful.
        The function should take an int (index), numpy.ndarray with points and a model as arguments.
        It should return a bool.

    Returns
    -------
    next_idx : int
        The row of the next point to cluster

    """

    for idx, point in enumerate(points):
        clustlib.cluster_point(point, model)
        if objective_fun(idx, points, model):
            return idx + 1
    return False


def predict_point(point, model, outlier_detection=False):
    """ Predicts which cluster a point belongs to.

    Parameters
    ----------
    point : numpy.ndarray

    model : bfr.Model

    outlier_detection : bool
        If True, outliers will be predicted with -1.
        If False, predictions will not consider default threshold.

    Returns
    -------
    closest_idx : int
        The index of the closest cluster (defined by default distance_fun).
        Returns -1 if the point is considered an outlier (determined by
        default threshold_fun and threshold)

        """

    closest_idx = clustlib.closest(point, model.discard, model.distance_fun)
    if not outlier_detection:
        return closest_idx
    if model.distance_fun(point, model.discard[closest_idx]) < model.threshold:
        return closest_idx
    return -1


def rss_error(points, model, outlier_detection=False):
    """ Compute the rss error of points given model. Optionally exclude outliers
    in the computation of the error.

    Parameters
    ----------
    points : numpy.ndarray
        (rows, dimensions) array with rows consisting of points. The points should
        have the same dimensionality as the model.

    model : bfr.Model

    outlier_detection : bool
        If True, outliers will not be considered when computing the error.
        If False, outliers will be considered when computing the error.

    Returns
    -------
    error : float
        The residual sum of square error of points. Computed using points and the centers
        in model.discard.

    """

    predictions = model.predict(points, outlier_detection)
    error = 0
    for idx, point in enumerate(points):
        prediction = predictions[idx]
        if not prediction == -1 and not ptlib.used(point):
            cluster = model.discard[prediction]
            error += clustlib.sum_squared_diff(point, cluster)
    return error
