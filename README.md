# VecLite-Db

VecLite-Db is a simple implementation of a vector database that uses SQLite for data storage.

## Description

VecLite-Db stores vectors in clusters, optimizing for efficient retrieval. The process involves:

- **Add into Store**: Creates clusters of embeddings and stores vectors according to their cluster.
- **Build a Kmean Tree**: Create a binary search tree on basis of clustering, first it create 2 cluster of whole data which will become left and right. Each node consists of 2 vectors, content, and metadata. Leaf node might have more than one  vectors..
   
  
- **Query**:
  - **Full Scan**: Calculates similarity with the query vector against all vectors.
  - **Cluster Scan**: Calculates similarity between the query vector and corresponding centroids, fetching data from the most similar cluster.
  - **Random Projection**: Reduces the dimension of vectors using Random Projection for improved storage and computational efficiency.
  - **KmeanTree**: Search approximate datapoints from kmean tree.


### How Kmean Tree Works:

#### Structure of Tree:
 ```
  tree = {
        "vectors":[],
        "content":"",
        "metadata":{},
        "left":{},
        "right":{}
        }
```

#### Algorithm:

- Start by creating two clusters for each node.

- Select two data points from the entire dataset that are closest to the centroids of the clusters.

- Split the data based on the clusters into left and right subsets.

- Repeat the same process recursively until the tree reaches a point where the number of data points is less than a predefined threshold.
