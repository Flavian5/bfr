"""This is a module defining bfr"""
import numpy


class Cluster:
    """A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes"""
    def __init__(self, dimensions):
        self.size = 0
        self.sums = numpy.zeros(dimensions)
        self.sums_sq = numpy.zeros(dimensions)


def std_dev(cluster):
    """ Computes the standard deviation within each dimension of a cluster.
    V(x) = E(x²) - (E(x))²
    sd(x) = sqrt(V(x))

    Parameters
    ----------
    cluster: namedtuple with the (int)size and numpy.ndarrays sums and sums_sq as attributes

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
    cluster: namedtuple with the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    mean (centroid):
    """
    return cluster.sums / cluster.size


def malahanobis(point, cluster):
    """ Computes the malahanobis distance between a cluster and a point.
    The malahanobis distance corresponds to the normalized Euclidean distance.
    Represents a likelihood that the point belongs to the cluster.

    Parameters
    ----------
    point : numpy.ndarray
    cluster : namedtuple with the (int)size and numpy.ndarrays sums and sums_sq as attributes

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
