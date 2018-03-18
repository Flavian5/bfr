# Clustering with BFR
## Synopsis
BFR is an algorithm for clustering large databases. <br />
BFR summarizes clusters using two main attributes:
* Sum in each dimension
* Sum of squares in each dimension
The sum and sum of squares allows efficient computation of the mean (centroid) and standard deviation. <br />
The cluster is represented by the centroid and spread (standard deviation) <br />
For each point added to a cluster, the sum and sum of squares are updated <br />
The BFR model contains three sets:
* The discard set
* The compress set
* The retain set
The discard set contains the main clusters. <br />
Points within a distance threshold of the closest centroid in the discard set will be included in that cluster. <br />
The compress set contains clusters of points which are far from main clusters but close to each other. <br />
The retain set contains points which are far from eachother and far from any main cluster. <br />
## Getting Started
git clone https://github.com/jeppeb91/bfr
### Prerequisites
If you are on a system supporting make: make init <br />
If you're system does not support make: pip install -r requirements.txt
### Running the tests
If you are on a system supporting make: make test <br />
If you're system does not support make: nosetests tests
### Coding style tests
If you are on a system supporting make: make lint <br />
If you're system does not support make: pylint ./bfr/*.py ./tests/*.py
## Contributing
Make a pull request and explain the whats and whys <br />
Catch me in person
## License
To be decided with Epidemic
## Acknowledgements
Bradley, Fayyad and Reina who suggested the approach <br />
[Link to the paper](https://www.aaai.org/Papers/KDD/1998/KDD98-002.pdf)



