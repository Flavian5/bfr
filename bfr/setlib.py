import bfr
from . import clustlib
from . import ptlib


def try_retain(point, model):
    new_cluster = clustlib.Cluster(model.dimensions)
    clustlib.update_cluster(point, new_cluster)
    if not model.retain:
        model.retain.append(new_cluster)
        return
    closest_idx = clustlib.closest(point, model.retain, clustlib.euclidean)
    closest_cluster = model.retain[closest_idx]
    if clustlib.euclidean(point, closest_cluster) < model.eucl_threshold:
        model.retain.pop(closest_idx)
        clustlib.update_cluster(point, closest_cluster)
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

    if ptlib.used(point):
        return True
    if not cluster_set:
        return False
    closest_idx = clustlib.closest(point, cluster_set, model.distance_fun)
    closest_cluster = cluster_set[closest_idx]
    if model.threshold_fun(point, closest_cluster) < model.threshold:
        clustlib.update_cluster(point, closest_cluster)
        return True
    return False


def finalize_set(clusters, model):
    for idx, cluster in enumerate(clusters):
        mean = clustlib.mean(cluster)
        closest_idx = clustlib.closest(mean, model.discard, model.distance_fun)
        closest_cluster = model.discard[closest_idx]
        merged = clustlib.merge_clusters(cluster, closest_cluster)
        model.discard[closest_idx] = merged


def update_compress(model):
    if len(model.compress) == 1:
        return
    for cluster in model.compress:
        clust = model.compress.pop(0)
        centroid = clustlib.mean(clust)
        closest_idx = clustlib.closest(centroid, model.compress, clustlib.mahalanobis)
        closest_cluster = model.compress[closest_idx]
        if clustlib.std_check(clust, closest_cluster, model.merge_threshold):
            merged = clustlib.merge_clusters(clust, closest_cluster)
            model.compress[closest_idx] = merged
            return update_compress(model)
        model.compress.append(clust)
