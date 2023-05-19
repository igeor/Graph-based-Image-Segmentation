import numpy as np 
import cv2 as cv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from create_graph import image_to_graph
from utils import * 
from core import *



def segmentation(image, sigma, neigh, K):
    # Create graph from image
    image, G = image_to_graph(image, k=neigh, sigma=sigma)

    # iterate over edges
    for edge in tqdm(sorted(G.edges(data=True), key=lambda x: x[2]['weight'])):
        # get u, v and weight
        u, v, weight = edge

        # get cluster of u and v
        cluster_u = get_cluster_of_node(G, u)
        cluster_v = get_cluster_of_node(G, v)

        # check if u and v are in the same cluster
        if cluster_u != cluster_v: 

            # get difference between clusters
            diff = diff_(G, cluster_u, cluster_v)
            # get minimum internal similarity of clusters
            mint = mint_(G, cluster_u, cluster_v, k=K)
            
            # print(f"diff: {round(diff, 3)} | mint: {round(mint, 3)}")
            if diff < mint: 
                # Merge clusters by updating the cluster of all nodes of cluster_2
                for node in get_nodes_of_cluster(G, cluster_v):
                    G.nodes[node]['cluster'] = cluster_u

    # For each cluster, create a set of nodes in that cluster
    clusters = []
    for cluster in set([G.nodes[node]['cluster'] for node in G.nodes()]):
        clusters.append(get_nodes_of_cluster(G, cluster))
    
    return G
    


    