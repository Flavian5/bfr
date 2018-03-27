# Clustering with BFR
BFR is a one pass algorithm for clustering large databases.

There are two main principles of the algorithm:
* Summarize points in clusters

* Make decisions which are likely directly. Push uncertain decisions into 
the future
## Synopsis
BFR summarizes clusters using two main attributes:
* Sum in each dimension
* Sum of squares in each dimension

The sum and sum of squares allows efficient computation of the mean (centroid) and standard deviation

The cluster is represented by the centroid and spread (standard deviation)

For each point added to a cluster, the sum and sum of squares are updated

The BFR model contains three sets:
* The discard set
* The compress set
* The retain set

The discard set contains the main clusters.

Points within a distance threshold of the closest cluster in the discard set will be included in that cluster.

Points outside of a distance threshold of the closest cluster in the discard set 
but within a distance threshold of the closest compress set cluster will be included in that compress set cluster.

Points outside of a distance threshold of both discard and compress set clusters will be included in the retain set.

If two points in the retain set are within a distance threshold they will be merged and moved to the compress set.

When the model has considered all points, the clusters in the compress and retain set get assigned to the closest cluster in discard.


## Code Examples
    import bfr
    import matplotlib.pyplot
    from sklearn.datasets.samples_generator import make_blobs
   
    vectors, _ = make_blobs(n_samples=1000, cluster_std=1,
                            n_features=dimensions, centers=nof_clusters,
                            shuffle=True)
                                   
    model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=3.0,
                      merge_threshold=2.0, dimensions=dimensions,
                      init_rounds=40, nof_clusters=nof_clusters)
    
    # Create the model using 500 vectors
    model.create(vectors[:500])
    
    # Update the model using 500 other vectors
    model.update(vectors[500:])
    
    # Finalize the model
    model.finalize()
    
    # Print the residual sum of square error
    print(model.error(vectors))
    
    # Predict the cluster of the points and plot
    predictions = self.model.predict(self.vectors, outlier_detection=True)
    x_cord, y_cord = self.vectors.T
    matplotlib.pyplot.scatter(x_cord, y_cord, c=predictions)
    matplotlib.pyplot.show()

## Getting Started
git clone https://github.com/jeppeb91/bfr
### Prerequisites
If you are on a system supporting make: make init

If you're system does not support make: pip install -r requirements.txt
### Running the tests
If you are on a system supporting make: make test

If you're system does not support make: nosetests tests
### Coding style tests
If you are on a system supporting make: make lint

If you're system does not support make: pylint ./bfr/*.py ./tests/*.py
## Contributing
Make a pull request and explain the whats and whys

Catch me in person
## License
To be decided with Epidemic
## Acknowledgements
Bradley, Fayyad and Reina who suggested the approach

[Link to the paper](https://www.aaai.org/Papers/KDD/1998/KDD98-002.pdf)
