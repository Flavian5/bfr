import sys
import traceback
import inspect
import numpy


def sanity_check(points, model, check_fun):
    try:
        check_fun(points, model)
    except AssertionError:
        _, _, trace_back = sys.exc_info()
        tb_info = traceback.extract_tb(trace_back)
        filename, line, func, text = tb_info[-1]
        method = inspect.stack()[2][3]
        sys.stderr.write("{} method: {}:".format(model, method))
        sys.stderr.write("\nProblem detected in {} in statement {}".format(func, text))
        return False
    return True


def confirm_attributes(model):
    return sanity_check('_', model, check_attributes)


def confirm_create(points, model):
    return sanity_check(points, model, check_create)


def confirm_update(points, model):
    return sanity_check(points, model, check_update)


def confirm_predict(points, model):
    return sanity_check(points, model, check_clusters)


def confirm_error(points, model):
    return sanity_check(points, model, check_clusters)


def check_attributes(_, model):
    assert isinstance(model.mahal_threshold, float), "mahalanobis_factor not float"
    assert isinstance(model.eucl_threshold, float), "euclidean_threshold not float"
    assert isinstance(model.merge_threshold, float), "merge_threshold not float"
    assert isinstance(model.init_rounds, int), "init_rounds not int"
    assert isinstance(model.dimensions, int), "dimensions not int"
    assert isinstance(model.nof_clusters, int), "nof_clusters not int"
    assert model.mahal_threshold > 0, "mahalanobis factor not > 0"
    assert model.eucl_threshold > 0, "euclidean_threshold not > 0"
    assert model.merge_threshold > 0, "merge_threshold not > 0"
    assert model.dimensions > 0, "dimensions not > 0"
    assert model.nof_clusters > 1, "nof_clusters not > 1"
    assert model.init_rounds > 0, "init_rounds not > 0"


def check_create(points, model):
    required_nr = model.nof_clusters * model.init_rounds
    check_points(points, model, required_nr)
    check_attributes('_', model)


def check_update(points, model):
    check_points(points, model, 1)
    check_attributes('_', model)


def check_clusters(points, model):
    check_points(points, model, 1)
    check_attributes('_', model)
    assert model.discard, "model has no clusters"
    if model.compress or model.retain:
        sys.stderr.write("Warning, you are predicting on a non finalized model. Expect less accuracy")


def check_points(points, model, required_nr):
    assert isinstance(points, numpy.ndarray), "Input points not numpy.ndarray"
    rows, dimensions = numpy.shape(points)
    assert dimensions == model.dimensions, "Dimension of points do not match model.dimensions"
    assert rows >= required_nr, "Not enough points"
