import numpy as np
import cv2 as cv 
import networkx as nx
from affinity import affinity

def image_to_graph(image, k=1, sigma=1.):
    
    # Define the Image Graph to return 
    G = nx.Graph()

    # Add pixels/nodes to the graph
    cluser_idx = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Define pixel index
            pixel_idx = (i, j)
            # Add node to graph
            G.add_node((i,j), rgb=image[i, j], cluster=cluser_idx)
            cluser_idx += 1 

    # Add edges between pixels that are within a k-hop neighborhood
    edges = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Add all pixels within a k x k window to the set of edges
            for r in range(max(0, i-k), min(image.shape[0], i+k+1)):
                for c in range(max(0, j-k), min(image.shape[1], j+k+1)):
                    # Prevent adding the same pixel as an edge
                    if r == i and c == j: continue
                    # Compute affinity between 2 pixels
                    affinity_value = affinity(image[i, j], image[r, c], sigma=sigma)
                    G.add_edge((i,j), (r,c), weight=affinity_value)

    return image, G