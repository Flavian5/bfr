# Clustering with BFR
BFR is an algorithm for clustering large databases

The algorithm requires only one pass over a dataset

The algorithm utilizes data compression to improve memory performance

There are two main principles of the algorithm:
* Summarize points in clusters. This results in a memory complexity which is O(clusters)

* Make decisions which are likely to be good directly. Push uncertain decisions into 
the future
![](https://i.imgur.com/ttatfNv.gif)
## Synopsis
BFR summarizes clusters using two main attributes:
* Sum in each dimension
* Sum of squares in each dimension

The sum and sum of squares allows efficient computation of the mean (centroid) and standard deviation

Clusters are represented by the centroid and spread (standard deviation)

The BFR model contains three sets:
* The discard set (blue, red)
* The compress set (gray)
* The retain set (black)

The <b>discard set</b> contains the main clusters.

The <b>compress set</b> contains points that are far from clusters in discard but close to eachother. These get summarized as clusters.

The <b>retain set</b> contains points that are far from clusters in both discard and compress and also far from other points in retain.

The initial step of the algorithm is to pick <b>k</b> points and let these be the main clusters in discard.

The outcome of the clustering is highly dependent on the <b>initial points</b>. Therefore the distance is maximized between the initial points with the following algorithm:
    
    let initial points = randomly picked point from input points
    while len(initial points) < k 
        Pick n candidates from input points at random
        choose the candidate that maximizes the distance to its closest point in initial points
        append the chosen candidate to initial points

After picking the initial point it is time to go throught the rest of the points

    foreach point in input points
        if two clusters in compress are close:
            merge them into one

        if point is within threshold to closest cluster in discard:
            update that cluster with point and continue loop
            
        if point is within threshold to closest cluster in compress:
            update that cluster with point and continue loop
        
        if point is within threshold to closest point in retain:
            combine those two points to a cluster and move the resulting cluster to compress
        
For each point added to a cluster, the <b>sum</b> and <b>sum of squares</b> of the cluster are updated. The point is then discarded.

When all points have been handled there are clusters in compress and points in retain. The final step of the algorithm is to assign/merge these to/with their closest cluster in discard. This is done in the <b>finallize()</b> call.

There are two distance functions: <b>Mahalanobis distance</b> and <b>Euclidean distance.</b>

<b>Euclidean distance</b> is the "normal" distance that most people think of.

<b>Mahalanobis distance</b> is a distance measure that considers the spread (standard deviation) of a cluster. The mahalanobis distance of a point and a cluster is lower if a cluster has a high spread. The mahalanobis distance of a point and a cluster is higher if a cluster has a low spread. The <b>mahalanobis distance is the main threshold function</b> in this implementation

<b>Euclidean distance</b> is used in the <b>initialization phase.</b> When clusters are represented by a point they have no spread and therefore mahalanobis distance may not be computed. The initialization phase is over when all discard clusters have a standard deviation in each dimension. <b>After the initalization phase mahalananobis distance is used</b> to determine nearness between a point and a cluster.

<b>Euclidean distance</b> is also used to determine if <b>two points in the retain set</b> are close to each other and should be merged and moved to compress. Euclidean distance is used for this also after the initialization phase has finished. This is because single points in discard do not have a spread.

Both Euclidean and mahalanobis distance have their own <b>corresponding threshold.</b>

<b>Merge threshold</b> is used to determine if two clusters in compress are considered close. Two clusters in compress will be merged if:

    std_dev(merged) < (std_dev(cluster) + std_dev(other)) * merge_threshold 

## Code Examples
    import bfr
    from sklearn.datasets.samples_generator import make_blobs
    
    # Generate test data
    vectors, _ = make_blobs(n_samples=1000, cluster_std=1,
                            n_features=2, centers=5,
                            shuffle=True)
              
    # Create the model. See below for parameter description                     
    model = bfr.Model(mahalanobis_factor=3.0, euclidean_threshold=3.0,
                      merge_threshold=2.0, dimensions=2,
                      init_rounds=40, nof_clusters=5)
    
    # Fit the model using 500 vectors
    model.fit(vectors[:500])
    
    # Fit (Update) the model using 500 other vectors
    model.fit(vectors[500:])
    
    # Finalize assigns clusters in the compress and retain set to the closest cluster in discard 
    model.finalize()
    
    # Print the residual sum of square error
    print(model.error(vectors))
    
    # Print cheap standard deviation based error without going through every vector
    print(model.error())
    
    # Predict the which cluster some points belong to
    predictions = model.predict(vectors[:2])

    # Predict the which cluster some points belong to
    # Outlier_detection=True defines that points far from their closest cluster will be identified
    predictions = model.predict(vectors[:2], outlier_detection=True)

    # Get the centers of the model as a numpy array
    centers = model.centers()
    
    # Print the model
    print(model)
    
    # Plot the model
    model.plot()
    
    # Plot the model and add points to the plot
    model.plot(vectors, outlier_detection=False)

## Model Attributes
mahalanobis_factor : float
        
    Nearness of point and cluster is determined by
    mahalanobis distance < mahalanobis_factor * sqrt(dimensions)
    Leaving this value at 3.0 usually provides good results.

euclidean_threshold : float
    
    Nearness when using euclidean distance is determined by
    Euclidean distance(point,cluster) < eucl_threshold
    This value has to be adapted depending on the range of euclidean distances between points.
 
merge_threshold : float
    
    Two clusters in the compress set will be merged if their merged standard deviation
    is less than or equal to (std_dev(cluster) + std_dev(other_cluster)) * merge_threshold.
    Keeping this value around 0.5 is usually a good starting point
 
dimensions : int
    
    The dimensionality of the model.
    This number should be equal to the number of features of points.

init_rounds : int
    
    Higher integer numbers give better spread of the initial points.
    Higher integer numbers give more stable results on the same dataset.
    Higher integer numbers increase the risk of starting with outlying points

nof_clusters : int
     
    The number of clusters (eg. K)

## Getting Started
git clone https://github.com/jeppeb91/bfr

Activate your virtual environment

    cd path/bfr
    
    make init (or pip install -r requirements.txt)
    
    pip install -e path/bfr
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
