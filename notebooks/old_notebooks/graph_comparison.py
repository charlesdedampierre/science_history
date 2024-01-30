import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from networkx.algorithms import similarity


def create_network(edges):
    """
    Creates a network from a list of edges.
    """
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)
    return G


def graph_edit_distance(G1, G2):
    """
    Calculate Graph Edit Distance between two networks.
    """
    return nx.graph_edit_distance(G1, G2)


def jaccard_similarity(G1, G2):
    """
    Calculate Jaccard Similarity for corresponding nodes in two networks.
    """
    # Assuming node correspondence is known and nodes are labeled from 0 to n-1
    jaccard_sims = []
    for node in G1.nodes():
        neighbors_G1 = set(G1.neighbors(node))
        neighbors_G2 = set(G2.neighbors(node))
        jaccard_sim = len(neighbors_G1.intersection(neighbors_G2)) / len(
            neighbors_G1.union(neighbors_G2)
        )
        jaccard_sims.append(jaccard_sim)
    return np.mean(jaccard_sims)


"""

Known node-correspondence (KNC) methods


1 - Weighted Jaccard distance (for weighted matrix) but not very used for method comparisons
2 - DeltaCon (The rationale of the method is that just measuring the overlap of the two edge sets might not work in practice, because not all edges have the same importance.)
3 - Cut Distance

Unknown node-correspondence (UNC) methods.

Subgraph Isomorphism
Spectral Analysis
alignment-based
graphlet-based
Portrait Divergence
NetLSD


Occupations are Known Node-Correspondence (KNC)


"""
