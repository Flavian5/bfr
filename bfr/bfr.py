"""This is a module defining bfr"""
from collections import namedtuple
import numpy

Cluster = namedtuple('Cluster', 'mean std_dev')


def init_cluster(dimensions):
    """
    :param dimensions: Number of dimensions of the standard deviation and mean vectors

    :return: Cluster with mean and standard deviation represented by zero vectors
    """
    mean = numpy.zeros(dimensions)
    std_dev = numpy.zeros(dimensions)
    return Cluster(mean, std_dev)
