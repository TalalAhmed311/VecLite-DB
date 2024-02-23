from veclite.vector_store_schema import VectorParams
from veclite.clsutering import Clustering
from veclite.database import SqliteDB
from veclite.distances import cosine_similarity
from typing import List
import numpy as np
import pickle


class VectorStore:

    def __init__(self,vector_dim: int = None,random_proj=False) -> None:

        self.clustering = Clustering(random_proj,vector_dim)
        self.sql_db = SqliteDB()

    

    def add(self,vectors_params: List[VectorParams]):


        if len(vectors_params)>30:

                num_clusters = np.random.randint(3,12)
                centroids, vector_models = self.clustering.create_cluster(vector_params=vectors_params,
                                                                        num_clusters=num_clusters)
                self.sql_db.add_centroids(centroids)
                self.sql_db.add_vectors(vector_models)


        else:
            centroid,vector_models = self.clustering.create_single_cluster(vector_params=vector_models)
            self.sql_db.add_centroids(centroid)
            self.sql_db.add_vectors(vector_models)

        

    def delete_all(self):
        self.sql_db.delete_all()

    def search_from_all(self,input_vector,top_k=20):
        results = self.sql_db.fetch_vectors()
        
        sim_vectors = []
        for vect in results:
            vector = pickle.loads(vect[2])

            sim_vectors.append(
                {
                    "cosine_sim":cosine_similarity(input_vector,vector),
                    "metadata":vect[3],
                    "content":vect[4]
            
                }
            )
            
        sim_vectors_sorted = sorted(sim_vectors, key=lambda x: x['cosine_sim'], reverse=True)
        for item in sim_vectors_sorted:
            item.pop('cosine_sim', None)


        return sim_vectors_sorted[:top_k]
        
    def search_from_cluster(self,input_vector,top_k=20):
        centroid_results = self.sql_db.fetch_centroids()

        sim_centroids = []

        for vect in centroid_results:
            vector = pickle.loads(vect[1])

            sim_centroids.append(
                {
                    "cosine_sim":cosine_similarity(input_vector,vector),
                    "id":vect[0]
            
                }
            )

        sim_centroids_sorted = sorted(sim_centroids, key=lambda x: x['cosine_sim'], reverse=True)


        results = []
        for cent in sim_centroids_sorted:
            sim_vectors = []
            result_vectors = self.sql_db.fetch_cluster_vectors(centroid_id=cent['id'])

            if len(result_vectors)<top_k:
                for vect in result_vectors:
                    vector = pickle.loads(vect[2])

                    sim_vectors.append(
                        {
                            "cosine_sim":cosine_similarity(input_vector,vector),
                            "metadata":vect[3]
                    
                        }
                    )
                    
                sim_vectors_sorted = sorted(sim_vectors, key=lambda x: x['cosine_sim'], reverse=True)
                for item in sim_vectors_sorted:
                    item.pop('cosine_sim', None)
                    results.append(item)
            else:
                for vect in result_vectors:
                    vector = pickle.loads(vect[2])

                    sim_vectors.append(
                        {
                            "cosine_sim":cosine_similarity(input_vector,vector),
                            "metadata":vect[3],
                            "content":vect[4]
                    
                        }
                    )
                    
                sim_vectors_sorted = sorted(sim_vectors, key=lambda x: x['cosine_sim'], reverse=True)
                for item in sim_vectors_sorted:
                    item.pop('cosine_sim', None)

                return sim_vectors_sorted[:top_k]
            

        return results[:top_k]

