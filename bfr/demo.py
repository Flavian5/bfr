def main():
    from bfr import bfr
    import matplotlib.pyplot
    from sklearn.datasets.samples_generator import make_blobs
    dimensions = 2
    nof_clusters = 5
    vectors, clusters = make_blobs(n_samples=1000, cluster_std=1,
                                   n_features=dimensions, centers=nof_clusters,
                                   shuffle=True)

    model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=3.0,
                      merge_threshold=2.0, dimensions=dimensions,
                      init_rounds=40, nof_clusters=nof_clusters)
    model.create(vectors)
    model.finalize()
    print(model.error(vectors))
    predictions = model.predict(vectors, outlier_detection=True)
    x_cord, y_cord = vectors.T
    matplotlib.pyplot.scatter(x_cord, y_cord, c=predictions)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    main()