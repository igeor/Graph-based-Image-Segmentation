import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv

def image_to_graph(image, k=1):
    """Convert image to graph
    Args:
        image: ndarray image to convert 
        k: neighborhood around each pixel
    Returns:
        vertices: vertices of graph
        edge_list: edge list of graph
    """
    # get the width and height of image
    width, height, c = image.shape
    # Create the graph
    G = nx.Graph()
    # Add nodes to the graph
    for i in range(width):
        for j in range(height):
            # Add node with feature vector
            G.add_node((i, j), feature=image[i, j])
    # Add edges to the graph
    # Iterate over all nodes
    for u in G.nodes():
        # get the neighbors of node u
        neighbors = []
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                if i == 0 and j == 0:
                    continue
                x = u[0] + i
                y = u[1] + j
                if x >= 0 and x < width and y >= 0 and y < height:
                    neighbors.append((x, y))
        # Add edges between u and its neighbors
        for v in neighbors:
            G.add_edge(u, v)
    # return graph
    return G     


if __name__ == '__main__':
    # load image with cv 
    image = cv.imread('./lena.png')
    # resize image
    image = cv.resize(image, (128, 128))
    # convert image to graph
    G = image_to_graph(image) 
    print('Number of nodes: ', G.number_of_nodes())
    print('Number of edges: ', G.number_of_edges())