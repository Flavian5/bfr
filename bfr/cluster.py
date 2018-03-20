import bfr
import numpy
from . import point_funs
from . import set_funs


class Cluster:
    """ A Cluster summarizes data of included points.

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
    cluster.has_variance = has_variance(cluster)


def closest(point, clusters, nearness_fun):
    """ Finds the cluster of which the centroid is closest to the point

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    clusters : list
        A list of clusters

    nearness_fun : function
        A distance function which accepts

    Returns
    -------
    cluster : The cluster with the closest mean (center)

    """

    dists = map(lambda cluster: nearness_fun(point, cluster), clusters)
    min_idx = numpy.argmin(list(dists))
    return min_idx


def merge_clusters(clust, other_cluster):
    """

    Parameters
    ----------
    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    other_cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    cluster : bfr.Cluster
        Cluster with updated sums, sums_sq, size and has_variance

    """
    dimensions = len(clust.sums)
    result = Cluster(dimensions)
    result.sums = clust.sums + other_cluster.sums
    result.sums_sq = clust.sums_sq + other_cluster.sums_sq
    result.has_variance = clust.has_variance & other_cluster.has_variance
    result.size = clust.size + other_cluster.size
    return result


def has_variance(cluster):
    """ Checks if a cluster has zero variance/std_dev in any dimension

    Parameters
    ----------
    cluster : bfr.Cluster

    Returns
    -------
    bool
        True if the cluster does not have 0 std_dev in any dimension, False otherwise

    """

    std_devs = std_dev(cluster)
    return numpy.all(std_devs)


def mean(cluster):
    """ Computes the mean of the cluster within each dimension

    Parameters
    ----------
    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    mean (centroid): numpy.ndarray
        The mean of each dimensions (the centroid)

    """

    return cluster.sums / cluster.size


def euclidean(point, cluster):
    """ Computes the euclidean distance between a point and the mean of a cluster
    d(v, w) = ||v - w|| = sqrt(sum(v_i - w_i)²)
    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    Euclidean distance : float

    """
    centroid = mean(cluster)
    return point_funs.euclidean(point, centroid)


def mahalanobis(point, cluster):
    """ Computes the mahalanobis distance between a cluster and a point.
    The mahalanobis distance corresponds to the normalized Euclidean distance.
    Represents a likelihood that the point belongs to the cluster.
    Note : If the variance is zero in any dimension, that dimension will be disregarded
    when computing the distance.

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    mahalanobis distance : float

    """

    diff = point - mean(cluster)
    std_devs = std_dev(cluster)
    if cluster.has_variance:
        normalized = diff / std_devs
    else:
        idx = numpy.where(std_devs != 0)
        normalized = diff[idx] / std_devs[idx]
    squared = normalized ** 2
    total = numpy.sum(squared)
    return numpy.sqrt(total)


def cluster_point(point, model):
    """ Update a model with point

    Parameters
    ----------
    point : numpy.ndarray
        The point to be clustered

    model : bfr.Model

    """

    if point_funs.used(point):
        return
    assigned = set_funs.try_include(point, model.discard, model)
    if not assigned:
        assigned = set_funs.try_include(point, model.compress, model)
    if not assigned:
        set_funs.try_retain(point, model)


def std_check(cluster, other_cluster, threshold):
    merged = merge_clusters(cluster, other_cluster)
    std = std_dev(merged)
    above_th = numpy.where(std > threshold)
    if numpy.any(above_th):
        return False
    return True


def std_dev(cluster):
    """ Computes the standard deviation within each dimension of a cluster.
    V(x) = E(x²) - (E(x))²
    sd(x) = sqrt(V(x))

    Parameters
    ----------
    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    standard deviation : numpy.ndarray
        The standard deviation of each dimension

    """

    expected_x2 = cluster.sums_sq / cluster.size
    expected_x = cluster.sums / cluster.size
    variance = expected_x2 - (expected_x ** 2)

    return numpy.sqrt(variance)