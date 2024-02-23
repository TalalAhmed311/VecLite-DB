import numpy as np
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
import uuid
import json
import pickle
from typing import List
from veclite.vector_store_schema import VectorParams

class Clustering:

    def __init__(self,vector_dim: int,random_proj=False) -> None:

        self.projector = GaussianRandomProjection(n_components=vector_dim//2)
        self.random_proj = False
    
    
    def create_cluster(self,vector_params: List[VectorParams],
                       num_clusters: int):

        vectors = [vector_param.vector for vector_param in vector_params]
        metadata = [vector_param.metadata for vector_param in vector_params]
        content = [vector_param.content for vector_param in vector_params]

        if self.random_proj:
            vectors = self.projector.fit_transform(vectors)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(vectors)

        cluster_labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        centroids = [(str(uuid.uuid4()),pickle.dumps(centroid)) for centroid in cluster_centroids]

        vector_models = []


        for idx,label in enumerate(cluster_labels):
            vector_models.append((
                str(uuid.uuid4()),
                centroids[label][0],
                pickle.dumps(vectors[idx]),
                content[idx],
                json.dumps(metadata[idx])
            ))

        return centroids, vector_models
    
    def create_single_cluster(self,vector_params: List[VectorParams]):

        vectors = [vector_param.vector for vector_param in vector_params]
        
        if self.random_proj:
            vectors = self.projector.fit_transform(vectors)

        cent_id = str(uuid.uuid4())
        centroid = [(cent_id,np.mean(vectors,axis=0))]
        vector_models = []

        for idx in range(len(vector_params)):

            vector_models.append({
                str(uuid.uuid4()),
                cent_id,
                pickle.dumps(vector_params[idx].vector),
                json.dumps(vector_params[idx].metadata)
            })

        return centroid, vector_models
            




