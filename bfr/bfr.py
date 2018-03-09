"""This is a module defining bfr"""
import numpy


class Cluster:
    """ A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes"""
    def __init__(self, dimensions):
        self.size = 0
        self.sums = numpy.zeros(dimensions)
        self.sums_sq = numpy.zeros(dimensions)


class Model:
    """

    Attributes
    ----------
    mahal_threshold : float
        Nearness of point and cluster is determined by mahalanobis distance < threshold * std_dev

    eucl_threshold : float
        Nearness of two points is determined by Euclidean distance < threshold

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
        self.mahal_threshold = kwargs.pop('mahalanobis_threshold', 3)
        self.eucl_threshold = kwargs.pop('euclidean_threshold', 1000)
        self.dimensions = kwargs.pop('dimensions')
        self.nof_clusters = kwargs.pop('nof_clusters')
        self.discard = [Cluster(self.dimensions) for i in range(self.nof_clusters)]
        self.compress = []
        self.retain = []

def update_cluster(point, cluster):
    """ Updates the given cluster according to the data of point

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------

    """
    cluster.size += 1
    cluster.sums += point
    cluster.sums_sq += point ** 2


def closest(point, clusters):
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

    dists = map(lambda cluster: euclidean(point, cluster), clusters)
    min_idx = numpy.argmin(list(dists))
    return clusters[min_idx]


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
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    malahanobis distance : float

    """

    diff = point - mean(cluster)
    normalized = diff / std_dev(cluster)
    #normalized = numpy.nan_to_num(normalized)
    squared = normalized ** 2
    total = numpy.sum(squared)
    return numpy.sqrt(total)
