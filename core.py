import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

from utils import get_nodes_of_cluster

""" 
Predicate D determines whether there is a boundary for segmentation.
"""

def diff_(G, cluster_1, cluster_2):
    """ 
    The difference between two clusters is the minimum weight edge 
    that connects a node u in cluster_1 to node v in cluster_2
    """
    # List the nodes of each cluster
    nodes_1 = get_nodes_of_cluster(G, cluster_1)
    nodes_2 = get_nodes_of_cluster(G, cluster_2)
    # Find all the edges between the 2 clusters
    edges_between_clusters = [(n1, n2) for n1 in nodes_1 for n2 in nodes_2 if G.has_edge(n1, n2)]
    # Find the minimum weight of the edges between the 2 clusters
    min_weight = min([G.edges[edge]['weight'] for edge in edges_between_clusters])
    return min_weight


def int_(G, cluster):
    """ 
    Internal similiarity of a cluster is the maximum weight edge 
    of the minimum spanning tree (MST) of the cluster.
    """
    # Compute minimum spanning tree of cluster_1 
    nodes_of_cluster = get_nodes_of_cluster(G, cluster)
    mst = nx.minimum_spanning_tree(G.subgraph(nodes_of_cluster))
    
    # The internal similarity of a singletons cluster is 0
    if len(mst.edges()) == 0: return 0

    # Find the maximum weight of the mst
    max_weight = max([mst.edges[edge]['weight'] for edge in mst.edges()])
    return max_weight

def t_(G, cluster, k=10):
    """ 
    t_ sets the threshold by which the clusters need to be different from
    the internal nodes in a cluster.
    Properties of constant k:
    ● If k is large, it causes a preference of larger objects.
    ● k does not set a minimum size for components.
    """
    n = len(get_nodes_of_cluster(G, cluster))
    return k / n

def mint_(G, cluster_1, cluster_2, k=10):
    """ 
    Minimum internal similarity between two clusters
    """
    return min(
        int_(G, cluster_1) + t_(G, cluster_1, k=k),
        int_(G, cluster_2) + t_(G, cluster_2, k=k)
    )