import random
import numpy

def sum_squared_diff(point, other_point):
    """ Computes the sum of dimensions of (point - other_point) ^ 2

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
    return numpy.dot(diff, diff)

def euclidean(point, other_point):
    """ Computes the euclidean distance between a point and another point
    d(v, w) = ||v - w|| = sqrt(sum(v_i - w_i)^2)

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    other_point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    Returns
    -------

    """
    sq_diff = sum_squared_diff(point, other_point)
    return numpy.sqrt(sq_diff)


def sum_all_euclideans(points):
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


def random_points(points, model, seed=None):
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
    for round_idx in range(model.init_rounds):
        idx = random.sample(range(max_index), model.nof_clusters)
        sample_points = points[idx]
        spread_score = sum_all_euclideans(sample_points)
        samples.append(idx)
        scores.append(spread_score)
    best_spread = numpy.argmax(scores)
    idx = samples[best_spread]
    initial_points = points[idx]
    points[idx] = numpy.nan
    return initial_points
