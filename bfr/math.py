"""This module contains mathematical functions used in bfr"""
import numpy
import bfr

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

def euclidean_points(point, other_point):
    """ Computes the euclidean distance between a point and another point
    d(v, w) = ||v - w|| = sqrt(sum(v_i - w_i)²)

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    other_point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    Returns
    -------

    """
    diff = point - other_point
    sum_squared = numpy.dot(diff, diff)
    return numpy.sqrt(sum_squared)


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
    return euclidean_points(point, centroid)


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
