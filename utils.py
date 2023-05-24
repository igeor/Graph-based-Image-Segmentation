import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def num_clusters(G):
    return len(set(nx.get_node_attributes(G, 'cluster').values()))

def get_nodes_of_cluster(G, cluster_idx):
    return [node for node in G.nodes() if G.nodes[node]['cluster'] == cluster_idx]

def get_cluster_of_node(G, pixel):
    return G.nodes[pixel]['cluster']

def get_clusters_labels(G):
    return list(set([G.nodes[node]['cluster'] for node in G.nodes()]))

def get_clusters(G):
    clusters = {}
    for cluster_idx in get_clusters_labels(G):
        clusters[cluster_idx] = get_nodes_of_cluster(G, cluster_idx)
    return clusters

def sort_weigts(G):
    # sort weights in a non-decreasing order   
    return sorted([G.edges[edge]['weight'] for edge in G.edges()])

def viz_segmentation(image, G, gray=False, display=False):
    # Map each cluster_idx to a color
    cluster_colors = {}
    # Find all different cluster labels of G nodes 
    cluster_labels = set(nx.get_node_attributes(G, 'cluster').values())
    # Define a color for each cluster label
    for cluster_label in cluster_labels:
        if gray:
            cluster_colors[cluster_label] = np.random.randint(0, 255)
        else:
            cluster_colors[cluster_label] = np.random.randint(0, 255, 3)
    # Create a new image with the same size as the original one
    segmented_image = np.zeros_like(image)
    # Iterate over all nodes of G
    for node in G.nodes():
        # Get the cluster of the node
        cluster = G.nodes[node]['cluster']
        # Set the pixel of the segmented image to the color of the cluster
        segmented_image[node[0], node[1]] = cluster_colors[cluster]

    if display:
        # Update the image on the second axis
        fix, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].imshow(image)
        axes[1].imshow(segmented_image)
        
    return segmented_image