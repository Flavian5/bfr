import bfr
from . import point_funs
from . import cluster
from . import objectives


def initialize(model, points, initial_points=None):
    """ Initializes clusters using points and optionally specified initial points.

    Parameters
    ----------
    points : numpy.matrix
        Matrix with rows consisting of points. The points should
        have the same dimensionality as the model.

    initial_points : numpy.matrix
        Matrix with rows that will be used as the initial centers.
        The points should have the same dimensionality as the model
        and the number of points should be equal to the number of clusters.

    Returns
    -------
    index : int
        Returns the row index to the first point not included in the model.
        Note : the point may be numpy.nan if it was randomly picked by
        random points.

    """

    if initial_points is None:
        initial_points = point_funs.random_points(model.nof_clusters, points, 100)
    initiate_clusters(initial_points, model)
    return cluster_points(points, model, objectives.zerofree_variances)


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
        clust = cluster.Cluster(model.dimensions)
        cluster.update_cluster(point, clust)
        model.discard.append(clust)


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
        cluster.cluster_point(point, model)
    return False


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
    closest_idx = cluster.closest(point, model.discard, model.distance_fun)
    if not outlier_detecion:
        return closest_idx
    if model.distance_fun(point, model.discard[closest_idx]) < model.threshold:
        return closest_idx
    return -1


def eucl_error(model, points, outlier_detection=False):
    predictions = model.predict(points, outlier_detection)
    centroids = list(map(lambda clust: cluster.mean(clust), model.retain))
    error = 0
    for idx, point in enumerate(points):
        prediction = predictions[idx]
        print("pred", point)
        print(error)
        if not prediction == -1 and not point_funs.used(point):
            clust = model.discard[int(prediction)]
            error += cluster.euclidean(point, clust)
    return error
