"""This is a module defining bfr"""
import random
import numpy


class Cluster:
    """ A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes"""
    def __init__(self, dimensions):
        self.size = 0
        self.sums = numpy.zeros(dimensions)
        self.sums_sq = numpy.zeros(dimensions)
        self.has_variance = False

class Model:
    """

    Attributes
    ----------
    mahal_threshold : float
        Nearness of point and cluster is determined by
        mahalanobis distance < threshold * sqrt(dimensions)

    eucl_threshold : float
        Nearness of two points is determined by Euclidean distance < threshold

    threshold : float
        The current default threshold used by the model

    threshold_fun : function
        The current default function for determining if a point and a cluster
        are considered close. The function should accept a point and a cluster
        and return a float.

    distance_fun : function
        The current default function for finding the closest cluster of a point.
        The function should accept a point and a cluster and return a float

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
        self.mahal_threshold = kwargs.pop('mahalanobis_factor', 2)
        self.eucl_threshold = kwargs.pop('euclidean_threshold', 3)
        self.threshold = self.eucl_threshold
        self.threshold_fun = euclidean
        self.distance_fun = euclidean
        self.dimensions = kwargs.pop('dimensions')
        self.nof_clusters = kwargs.pop('nof_clusters')
        self.mahal_threshold = self.mahal_threshold * numpy.sqrt(self.dimensions)
        self.discard = []
        self.compress = []
        self.retain = []

    def initialize(self, points, initial_points=None):
        """ Initializes clusters using

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
            initial_points = random_points(self.nof_clusters, points)
        initiate_clusters(initial_points, self)
        return cluster_points(points, self, zerofree_variances)

    def update_model(self, points, idx=0):
        """

        Parameters
        ----------
        points : numpy.matrix
            Matrix with rows consisting of points. The points should
            have the same dimensionality as the model.

        idx : int
            The first row index from which the model will be updated.

        Returns
        -------
        bool
            True if the clustering clusters all points. False otherwise
        """
        next_idx = cluster_points(points[idx:], self, finish_points)
        if next_idx:
            return True
        return False

    def create(self, points, initial_points=None):
        """ Creates a bfr model from points using the initial points specified in
        initial points.

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
        bool
            True if the model was succesfully created. False otherwise.
        """
        if self.initialize(points, initial_points):
            self.threshold_fun = mahalanobis
            self.threshold = self.mahal_threshold
            self.update_model(points)
            return True
        return False

    def predict(self, point):
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

        closest_cluster = closest(point, self.discard, mahalanobis)
        idx = self.discard.index(closest_cluster)
        if self.distance_fun(point, closest_cluster) < self.threshold:
            return idx
        else:
            return -1


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


def initiate_clusters(initial_points, model):
    """

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
        model.retain.append(point)


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
    closest_cluster = closest(point, cluster_set, model.distance_fun)
    if model.threshold_fun(point, closest_cluster) < model.threshold:
        update_cluster(point, closest_cluster)
        return True
    return False


def zerofree_variances(_, __, model):
    """

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


def random_points(nof_points, points, seed=None):
    """ Returns a number of random points from points. Marks the selected points
    as used by setting them to numpy.nan

    Parameters
    ----------
    nof_points : int
        The number of points to be returned

    points : numpy.matrix
        The points from which the random points will be picked.
        Randomly picked points will be set to numpy.nan

    seed : int
        Sets the random seed

    Returns
    -------
    initial points : numpy.matrix
        The randomly chosen points
    """
    max_index = len(points) - 1
    random.seed(seed)
    idx = random.sample(range(max_index), nof_points)
    initial_points = points[idx]
    points[idx] = numpy.nan
    return initial_points


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
    cluster.has_variance = has_variance(cluster)


def closest(point, clusters, nearness_fun):
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

    dists = map(lambda cluster: nearness_fun(point, cluster), clusters)
    min_idx = numpy.argmin(list(dists))
    return clusters[min_idx]


def has_variance(cluster):
    """ Checks if a cluster has zero variance/std_dev in any dimension

    Parameters
    ----------
    cluster : bfr.Cluster

    Returns
    -------
    bool
        True if the cluster does not have 0 std_dev in any dimension, False otherwise
    """
    std_devs = std_dev(cluster)
    return numpy.all(std_devs)


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


def mahalanobis(point, cluster):
    """ Computes the mahalanobis distance between a cluster and a point.
    The mahalanobis distance corresponds to the normalized Euclidean distance.
    Represents a likelihood that the point belongs to the cluster.

    Parameters
    ----------
    point : numpy.ndarray
        Vector with the same dimensionality as the bfr model

    cluster : bfr.Cluster
        A cluster has the (int)size and numpy.ndarrays sums and sums_sq as attributes

    Returns
    -------
    mahalanobis distance : float

    """

    diff = point - mean(cluster)
    std_devs = std_dev(cluster)
    if cluster.has_variance:
        normalized = diff / std_devs
    else:
        idx = numpy.where(std_devs != 0)
        normalized = diff[idx] / std_devs[idx]
    squared = normalized ** 2
    total = numpy.sum(squared)
    return numpy.sqrt(total)
