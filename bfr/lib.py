"""Contains helper functions for bfr"""
import random
import numpy
import bfr


class Cluster:
    """

    Attributes
    ----------
    sums : numpy.ndarray
        Total sum of each dimension of the cluster.

    sums_sq : numpy.ndarray
        The sum of squares within each dimension

    has_variance : bool
        A boolean flag which is False when a cluster has zero variance in any dimension

    """
    def __init__(self, dimensions):
        self.size = 0
        self.sums = numpy.zeros(dimensions)
        self.sums_sq = numpy.zeros(dimensions)
        self.has_variance = False


def initiate_clusters(initial_points, model):
    """

    Parameters
    ----------
    initial_points : numpy.matrix
        Matrix with rows that will be used as the initial centers.
        The points should have the same dimensionality as the model
        and the number of points should be equal to the number of clusters.

    model : bfr.Model

    """

    for point in initial_points:
        cluster = Cluster(model.dimensions)
        update_cluster(point, cluster)
        model.discard.append(cluster)


def cluster_points(points, model, objective_fun):
    """ Update a model with the given points. Finished when objective is reached.

    Parameters
    ----------
    points : numpy.matrix
        Matrix with rows consisting of points. The points should
        have the same dimensionality as the model.

    model : bfr.Model

    objective_fun : function
        The objective_fun determines when the clustering has been succesful.
        The function should take an int (index), the points and a model as arguments
        and return a bool.

    Returns
    -------
    bool
        True if objective is reached, False otherwise.

    """

    for idx, point in enumerate(points):
        if objective_fun(idx, points, model):
            return idx + 1
        cluster_point(point, model)
    return False


def cluster_point(point, model):
    """ Update a model with point

    Parameters
    ----------
    point : numpy.ndarray
        The point to be clustered

    model : bfr.Model

    """

    if used(point):
        return
    assigned = try_include(point, model.discard, model)
    if not assigned:
        assigned = try_include(point, model.compress, model)
    if not assigned:
        model.retain.append(point)


def try_include(point, cluster_set, model):
    """ Includes a point in the closest cluster of cluster_set if it is considered close.
    Distance, threshold and threshold function is given by the defaults of model.

    Parameters
    ----------
    point : numpy.ndarray
        The point to try

    cluster_set : list
        the list of clusters to try

    model : bfr.model
        Default threshold and threshold_fun settings of the model determine if
        a point will be included with the set.

    Returns
    -------
    bool
        True if the point is assigned to a cluster. False otherwise

    """

    if used(point):
        return True
    if not cluster_set:
        return False
    closest_cluster = closest(point, cluster_set, model.distance_fun)
    if model.threshold_fun(point, closest_cluster) < model.threshold:
        update_cluster(point, closest_cluster)
        return True
    return False


def random_points(nof_points, points, seed=None):
    """ Returns a number of random points from points. Marks the selected points
    as used by setting them to numpy.nan

    Parameters
    ----------
    nof_points : int
        The number of points to be returned

    points : numpy.matrix
        The points from which the random points will be picked.
        Randomly picked points will be set to numpy.nan

    seed : int
        Sets the random seed

    Returns
    -------
    initial points : numpy.matrix
        The randomly chosen points
    """
    max_index = len(points) - 1
    random.seed(seed)
    idx = random.sample(range(max_index), nof_points)
    initial_points = points[idx]
    points[idx] = numpy.nan
    return initial_points


def used(point):
    """

    Parameters
    ----------
    point : numpy.ndarray

    Returns
    -------
    bool
        True if the point has been used. Represented by row being numpy.nan
    """
    return numpy.isnan(point[0])


def update_cluster(point, cluster):
    """ Updates the given cluster according to the data of point

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    """
    cluster.size += 1
    cluster.sums += point
    cluster.sums_sq += point ** 2
    cluster.has_variance = bfr.has_variance(cluster)


def closest(point, clusters, nearness_fun):
    """ Finds the cluster of which the centroid is closest to the point

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    clusters : list
        A list of clusters

    Returns
    -------
    cluster : The cluster with the closest mean (center)

    """

    dists = map(lambda cluster: nearness_fun(point, cluster), clusters)
    min_idx = numpy.argmin(list(dists))
    return clusters[min_idx]
