import random
import numpy


def euclidean(point, other_point):
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


def deltas(points):
    total = 0
    for point in points:
        diffs = points - point
        squared = diffs ** 2
        summed = numpy.sum(squared)
        total += numpy.sqrt(summed)
    return total


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


def random_points(nof_points, points, rounds, seed=None):
    """ Returns a number of random points from points. Marks the selected points
    as used by setting them to numpy.nan

    Parameters
    ----------
    nof_points : int
        The number of points to be returned

    points : numpy.matrix
        The points from which the random points will be picked.
        Randomly picked points will be set to numpy.nan

    rounds : int
        The number of initialization rounds

    seed : int
        Sets the random seed

    Returns
    -------
    initial points : numpy.matrix
        The round of random points which maximizes the distance between all

    """

    max_index = len(points) - 1
    random.seed(seed)
    samples = []
    scores = []
    for round_idx in range(rounds):
        idx = random.sample(range(max_index), nof_points)
        sample_points = points[idx]
        delta_sum = deltas(sample_points)
        samples.append(idx)
        scores.append(delta_sum)
    best_spread = numpy.argmax(scores)
    idx = samples[best_spread]
    initial_points = points[idx]
    points[idx] = numpy.nan
    return initial_points
