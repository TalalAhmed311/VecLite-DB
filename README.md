# VecLite-Db

VecLite-Db is a simple implementation of a vector database that uses SQLite for data storage.

## Description

VecLite-Db stores vectors in clusters, optimizing for efficient retrieval. The process involves:

- **Add into Store**: Creates clusters of embeddings and stores vectors according to their cluster.

- **Query**:
  - **Full Scan**: Calculates similarity with the query vector against all vectors.
  - **Cluster Scan**: Calculates similarity between the query vector and corresponding centroids, fetching data from the most similar cluster.
  - **Random Projection**: Reduces the dimension of vectors using Random Projection for improved storage and computational efficiency.
