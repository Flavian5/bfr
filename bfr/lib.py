"""Contains helper functions for bfr"""
import random
import numpy
import bfr


class Cluster:
    """ A Cluster summarizes data of included points.

    Attributes
    ----------
    sums : numpy.ndarray
        Total sum of each dimension of the cluster.

    sums_sq : numpy.ndarray
        The sum of squares within each dimension

    has_variance : bool
        A boolean flag which is False when a cluster has zero variance in any dimension

    """
    def __init__(self, dimensions):
        self.size = 0
        self.sums = numpy.zeros(dimensions)
        self.sums_sq = numpy.zeros(dimensions)
        self.has_variance = False


def initiate_clusters(initial_points, model):
    """ Updates the model with the initial cluster centers specified in initial_points.

    Parameters
    ----------
    initial_points : numpy.matrix
        Matrix with rows that will be used as the initial centers.
        The points should have the same dimensionality as the model
        and the number of points should be equal to the number of clusters.

    model : bfr.Model

    """

    for point in initial_points:
        cluster = Cluster(model.dimensions)
        update_cluster(point, cluster)
        model.discard.append(cluster)


def cluster_points(points, model, objective_fun):
    """ Update a model with the given points. Finished when objective is reached.

    Parameters
    ----------
    points : numpy.matrix
        Matrix with rows consisting of points. The points should
        have the same dimensionality as the model.

    model : bfr.Model

    objective_fun : function
        The objective_fun determines when the clustering has been succesful.
        The function should take an int (index), the points and a model as arguments
        and return a bool.

    Returns
    -------
    bool
        True if objective is reached, False otherwise.

    """

    for idx, point in enumerate(points):
        if objective_fun(idx, points, model):
            return idx + 1
        cluster_point(point, model)
    return False


def cluster_point(point, model):
    """ Update a model with point

    Parameters
    ----------
    point : numpy.ndarray
        The point to be clustered

    model : bfr.Model

    """

    if used(point):
        return
    assigned = try_include(point, model.discard, model)
    if not assigned:
        assigned = try_include(point, model.compress, model)
    if not assigned:
        try_retain(point, model)


def try_retain(point, model):
    new_cluster = bfr.Cluster(model.dimensions)
    update_cluster(point, new_cluster)
    if not model.retain:
        model.retain.append(new_cluster)
        return
    closest_idx = closest(point, model.retain, bfr.euclidean)
    closest_cluster = model.retain[closest_idx]
    if bfr.euclidean(point, closest_cluster) < model.eucl_threshold:
        model.retain.pop(closest_idx)
        bfr.update_cluster(point, closest_cluster)
        model.compress.append(closest_cluster)
    else:
        model.retain.append(new_cluster)


def try_include(point, cluster_set, model):
    """ Includes a point in the closest cluster of cluster_set if it is considered close.
    Distance, threshold and threshold function is given by the defaults of model.

    Parameters
    ----------
    point : numpy.ndarray
        The point to try

    cluster_set : list
        the list of clusters to try

    model : bfr.model
        Default threshold and threshold_fun settings of the model determine if
        a point will be included with the set.

    Returns
    -------
    bool
        True if the point is assigned to a cluster. False otherwise

    """

    if used(point):
        return True
    if not cluster_set:
        return False
    closest_idx = closest(point, cluster_set, model.distance_fun)
    closest_cluster = cluster_set[closest_idx]
    if model.threshold_fun(point, closest_cluster) < model.threshold:
        update_cluster(point, closest_cluster)
        return True
    return False


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


def update_cluster(point, cluster):
    """ Updates the given cluster according to the data of point

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    """
    cluster.size += 1
    cluster.sums += point
    cluster.sums_sq += point ** 2
    cluster.has_variance = bfr.has_variance(cluster)


def closest(point, clusters, nearness_fun):
    """ Finds the cluster of which the centroid is closest to the point

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    clusters : list
        A list of clusters

    nearness_fun : function
        A distance function which accepts

    Returns
    -------
    cluster : The cluster with the closest mean (center)

    """

    dists = map(lambda cluster: nearness_fun(point, cluster), clusters)
    min_idx = numpy.argmin(list(dists))
    return min_idx


def merge_clusters(cluster, other_cluster):
    """

    Parameters
    ----------
    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    other_cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    cluster : bfr.Cluster
        Cluster with updated sums, sums_sq, size and has_variance

    """
    dimensions = len(cluster.sums)
    result = bfr.Cluster(dimensions)
    result.sums = cluster.sums + other_cluster.sums
    result.sums_sq = cluster.sums_sq + other_cluster.sums_sq
    result.has_variance = cluster.has_variance & other_cluster.has_variance
    result.size = cluster.size + other_cluster.size
    return result


def std_check(cluster, other_cluster, threshold):
    merged = merge_clusters(cluster, other_cluster)
    std = bfr.std_dev(merged)
    above_th = numpy.where(std > threshold)
    if numpy.any(above_th):
        return False
    return True


def finalize_set(clusters, model):
    for idx, cluster in enumerate(clusters):
        mean = bfr.mean(cluster)
        closest_idx = bfr.closest(mean, model.discard, model.distance_fun)
        closest_cluster = model.discard[closest_idx]
        model.discard[closest_idx] = bfr.merge_clusters(cluster, closest_cluster)


def update_compress(model, threshold):
    if len(model.compress) == 1:
        return
    for cluster in model.compress:
        cluster = model.compress.pop(0)
        centroid = bfr.mean(cluster)
        closest_idx = closest(centroid, model.compress, bfr.mahalanobis)
        closest_cluster = model.compress[closest_idx]
        if std_check(cluster, closest_cluster, model.merge_threshold):
            model.compress.pop(closest_idx)
            merged = bfr.merge_clusters(cluster, closest_cluster)
            model.compress.append(merged)
            return update_compress(model, threshold)
        model.compress.append(closest_cluster)


def predict_point(point, model, outlier_detecion=False):
    """ Predicts which cluster a point belongs to.

    Parameters
    ----------
    point : numpy.ndarray
        The point to be predicted

    Returns
    -------
    idx : int
        The index of the closest cluster (defined by default distance_fun).
        Returns -1 if the point is considered an outlier (determined by
        default threshold_fun and threshold)

        """
    closest_idx = bfr.closest(point, model.discard, model.distance_fun)
    if not outlier_detecion:
        return closest_idx
    if model.distance_fun(point, model.discard[closest_idx]) < model.threshold:
        return closest_idx
    return -1

def eucl_error(model, points, outlier_detection=False):
    predictions = model.predict(points, outlier_detection)
    centroids = list(map(lambda cluster: bfr.mean(cluster), model.retain))
    error = 0
    for idx, point in enumerate(points):
        prediction = predictions[idx]
        print("pred", point)
        print(error)
        if not prediction == -1 and not used(point):
            cluster = model.discard[int(prediction)]
            error += bfr.euclidean(point, cluster)
    return error
