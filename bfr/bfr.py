"""This is a module defining bfr"""
from collections import namedtuple
import numpy

Cluster = namedtuple('Cluster', 'mean std_dev')


def init_cluster(dimensions):
    """Initializes a clusterr

    Parameters
    ----------
    dimensions : Dimensionality of the cluster

    Returns
    -------
    Cluster : namedtuple with the numpy arrays mean and std_dev as attributes

    """
    mean = numpy.zeros(dimensions)
    std_dev = numpy.zeros(dimensions)
    return Cluster(mean, std_dev)


def malahanobis(point, cluster):
    """ Computes the malahanobis distance between a cluster and a point.
    The malahanobis distance corresponds to the normalized Euclidean distance.
    Represents a likelihood that the point belongs to the cluster.

    Parameters
    ----------
    point : numpy array with dimensionality N
    cluster : Named tuple with the numpy arrays mean and std_dev as attributes

    Returns
    -------
    malahanobis distance : float

    """
    diff = point - cluster.mean
    normalized = diff / cluster.std_dev
    squared = normalized ** 2
    total = numpy.sum(squared)
    return numpy.sqrt(total)
