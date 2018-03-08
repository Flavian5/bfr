"""This is a module defining bfr"""
import numpy


class Cluster:
    """A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes"""
    def __init__(self, dimensions):
        self.size = 0
        self.sums = numpy.zeros(dimensions)
        self.sums_sq = numpy.zeros(dimensions)


def closest(point, clusters):
    """ Finds the cluster of which the centroid is closest to the point

    Parameters
    ----------
    point : numpy.ndarray
    clusters : numpy.ndarray of Clusters

    Returns
    -------
    cluster : The cluster with the closest mean (center)
    """
    eucl = numpy.vectorize(lambda cluster: euclidean(point, cluster))
    distances = eucl(clusters)
    min_idx = numpy.argmin(distances)
    return clusters[min_idx]


def std_dev(cluster):
    """ Computes the standard deviation within each dimension of a cluster.
    V(x) = E(x²) - (E(x))²
    sd(x) = sqrt(V(x))

    Parameters
    ----------
    cluster : Cluster with the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    standard deviation vector: numpy.ndarray with the standard deviation of each dimension

    """

    expected_x2 = cluster.sums_sq / cluster.size
    expected_x = cluster.sums / cluster.size
    variance = expected_x2 - (expected_x ** 2)
    return numpy.sqrt(variance)


def mean(cluster):
    """

    Parameters
    ----------
    cluster : Cluster with the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    mean (centroid):
    """
    return cluster.sums / cluster.size


def euclidean(point, cluster):
    """ Computes the euclidean distance between a point and the mean of a cluster
    d(v, w) = ||v - w|| = sqrt(sum(v_i - w_i)²)
    Parameters
    ----------
    point : numpy.ndarray
    cluster : Cluster with the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    Euclidean distance : float

    """
    diff = point - mean(cluster)
    sum_squared = numpy.dot(diff, diff)
    return numpy.sqrt(sum_squared)


def malahanobis(point, cluster):
    """ Computes the malahanobis distance between a cluster and a point.
    The malahanobis distance corresponds to the normalized Euclidean distance.
    Represents a likelihood that the point belongs to the cluster.

    Parameters
    ----------
    point : numpy.ndarray
    cluster : Cluster with the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    malahanobis distance : float

    """
    diff = point - mean(cluster)
    normalized = diff / std_dev(cluster)
    normalized = numpy.nan_to_num(normalized)
    squared = normalized ** 2
    total = numpy.sum(squared)
    return numpy.sqrt(total)
