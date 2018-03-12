"""This module contains objective functions."""


def finish_points(idx, points, _):
    """ Used to determine when all points have been clustered.

    Parameters
    ----------
    idx : int

    points : numpy.matrix

    Returns
    -------
    bool
        True if idx is the last row index of points. False otherwise.

    """

    return idx == len(points) - 1


def zerofree_variances(_, __, model):
    """ Used to determine when all clusters of the retain set have non zero variance
    in all dimensions

    Parameters
    ----------
    model : bfr.model

    Returns
    -------
    bool
        True if all clusters within the discard set of model has a non zero variance.
        False otherwise

    """

    has_variances = filter(lambda cluster: cluster.has_variance, model.discard)
    return len(list(has_variances)) == model.nof_clusters
