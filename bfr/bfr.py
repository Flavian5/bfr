"""This is a module defining bfr"""
from collections import namedtuple
import numpy

Cluster = namedtuple('Cluster', 'mean std_dev')


def init_cluster(dimensions):
    """

    Parameters
    ----------
    dimensions: Dimensionality of the cluster

    Returns
    -------
    namedtuple with the attributes mean and std_dev

    """
    mean = numpy.zeros(dimensions)
    std_dev = numpy.zeros(dimensions)
    return Cluster(mean, std_dev)
