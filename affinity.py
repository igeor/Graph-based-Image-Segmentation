import numpy as np 
import cv2 as cv 
from create_graph import image_to_graph

def affinity(G, u, v, sigma=1.):
    # get the feature vector of node u
    f_u = G.nodes[u]['feature']
    # get the feature vector of node v
    f_v = G.nodes[v]['feature']
    # compute the affinity between u and v
    return np.exp(
        (-np.linalg.norm(f_u - f_v)**2) / (2 * sigma**2)
        )

