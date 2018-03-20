import bfr
from . import cluster
from . import point_funs


def try_retain(point, model):
    new_cluster = cluster.Cluster(model.dimensions)
    cluster.update_cluster(point, new_cluster)
    if not model.retain:
        model.retain.append(new_cluster)
        return
    closest_idx = cluster.closest(point, model.retain, cluster.euclidean)
    closest_cluster = model.retain[closest_idx]
    if cluster.euclidean(point, closest_cluster) < model.eucl_threshold:
        model.retain.pop(closest_idx)
        cluster.update_cluster(point, closest_cluster)
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

    if point_funs.used(point):
        return True
    if not cluster_set:
        return False
    closest_idx = cluster.closest(point, cluster_set, model.distance_fun)
    closest_cluster = cluster_set[closest_idx]
    if model.threshold_fun(point, closest_cluster) < model.threshold:
        cluster.update_cluster(point, closest_cluster)
        return True
    return False


def finalize_set(clusters, model):
    for idx, clust in enumerate(clusters):
        mean = cluster.mean(clust)
        closest_idx = cluster.closest(mean, model.discard, model.distance_fun)
        closest_cluster = model.discard[closest_idx]
        model.discard[closest_idx] = cluster.merge_clusters(clust, closest_cluster)


def update_compress(model, threshold):
    if len(model.compress) == 1:
        return
    for clust in model.compress:
        clust = model.compress.pop(0)
        centroid = cluster.mean(clust)
        closest_idx = cluster.closest(centroid, model.compress, cluster.mahalanobis)
        closest_cluster = model.compress[closest_idx]
        if cluster.std_check(clust, closest_cluster, model.merge_threshold):
            model.compress.pop(closest_idx)
            merged = cluster.merge_clusters(clust, closest_cluster)
            model.compress.append(merged)
            return update_compress(model, threshold)
        model.compress.append(closest_cluster)
