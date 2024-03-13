import pandas as pd
import numpy as np
from typing import List

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from veclite.distances import cosine_similarity
from veclite.vector_store_schema import VectorParams
import os
import pickle

class KmeanTree:

    def create_tree(self,vector_params: List[VectorParams]):

    
        if len(vector_params)<=10:
            return {

                "vectors":[param.vector for param in vector_params],
                "content":[param.content for param in vector_params],
                "metadata":[param.metadata for param in vector_params],
                "left":{},
                "right":{}

            }

        X = [param.vector for param in vector_params]
        
        # Apply KMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)

        # Get cluster labels and centroids
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    
        # select 2 close points from each cluster
        closest_points_indices, _ = pairwise_distances_argmin_min(X, centroids,axis=0)

        vectors = [X[idx] for idx in closest_points_indices]
        content_list = [vector_params[idx].content for idx in closest_points_indices]
        metadata_list = [vector_params[idx].metadata for idx in closest_points_indices]

        tree = {
        "vectors":vectors,
        "content":"",
        "metadata":{},
        "left":{},
        "right":{}
        }

        # tree['vectors'] = vectors
        tree['content'] = content_list
        tree['metadata'] = metadata_list


        vector_params = np.delete(vector_params,closest_points_indices,axis=0)
        labels = np.delete(labels,closest_points_indices,axis=0)

        cluster_left = vector_params[labels == 0]
        cluster_right = vector_params[labels == 1]


        tree['left'] = self.create_tree(cluster_left)
        tree['right'] = self.create_tree(cluster_right)


        return tree

    def upsert(self,tree,vector_params: List[VectorParams]):

        left_tree = tree
        right_tree = self.create_tree(vector_params=vector_params)

        root_tree = {
            "vectors":[left_tree['vectors'][0], right_tree['vectors'][1]],
            "content":[left_tree['content'][0], right_tree['content'][1]],
            "metadata":[left_tree['metadata'][0], right_tree['metadata'][1]],
            "left":left_tree,
            "right":right_tree
        }

        return root_tree
        


    def format_tree(self,content_list,metadata_list):
        results = []
        for content, metadata in zip(content_list,metadata_list):
            results.append({"content":content,"metadata":metadata})
        
        return results

    def search(self,tree,point,array,top_k=20):

        if len(array)>=top_k:
            return array

        if not tree:
            return array 
        

        # condition to handle leaf node
        if len(tree['vectors'])<2 or len(tree['vectors'])>2:
            array.extend(self.format_tree(tree['content'],tree['metadata']))
            return array



        dist = [cosine_similarity(point,node) for node in tree['vectors']]
        
        if dist[0]>dist[1]:
            array.extend(self.format_tree(tree['content'],tree['metadata']))
            array= self.search(tree=tree['left'],point=point,array=array,top_k=top_k)

            if len(array)<top_k:
                array = self.search(tree=tree['right'],point=point,array=array,top_k=top_k)

        else:
            array.extend(self.format_tree(tree['content'],tree['metadata']))
            
            array = self.search(tree=tree['right'],point=point,array=array,top_k=top_k)

            if len(array)<top_k:
                array = self.search(tree=tree['left'],point=point,array=array,top_k=top_k)


        return array
    

    def save_tree(self,tree,file_name = 'tree.pickle'):

        dir_path = os.path.join(os.getcwd(),'vector_tree')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        
        file_dest = os.path.join(dir_path,file_name)

        if os.path.exists(file_dest):
            try:
                os.remove(file_dest)
                print(f"File '{file_name}' deleted successfully.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"File '{file_name}' does not exist in the directory.")

            

        with open(file_dest, 'wb') as pickle_file:
                pickle.dump(tree, pickle_file)
            

