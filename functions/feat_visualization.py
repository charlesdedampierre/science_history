import community.community_louvain as community
import networkx as nx
import numpy as np
import pandas as pd
from ipysigma import Sigma
from sklearn.preprocessing import MinMaxScaler
import leidenalg as la
import igraph
from igraph import Graph


def sygma_graph_leiden(
    df_edges: pd.DataFrame,
    df_nodes: pd.DataFrame,
    filepath: str,
    edge_bins: int = 10,
    node_bins: int = 10,
):
    df_edges["weight"] = pd.qcut(
        df_edges["weight"].rank(method="first"), edge_bins, np.arange(1, edge_bins + 1)
    ).astype(int)
    df_edges = df_edges[["source", "target", "weight"]].reset_index(drop=True)

    # Create Graph Object
    g = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", edge_attr="weight"
    )
    ig_graph = Graph(directed=False)
    g_ig = ig_graph.from_networkx(g)
    part = la.find_partition(
        g_ig, la.ModularityVertexPartition, n_iterations=100, seed=42
    )

    node_names = []
    indexes = []

    for element in g_ig.vs:
        node_names.append(element["_nx_name"])
        indexes.append(element.index)

    df_elements = pd.DataFrame({"element_id": indexes, "node": node_names})

    communities = []
    for community_nodes in part:
        community = set(community_nodes)
        communities.append(community)

    clusters_list = list(part)

    # Create a DataFrame with 'element_id' and 'cluster_id' columns
    df_partition = pd.DataFrame(
        [
            (element_id, cluster_id)
            for cluster_id, elements in enumerate(clusters_list)
            for element_id in elements
        ],
        columns=["element_id", "community"],
    )

    df_partition = pd.merge(df_partition, df_elements, on="element_id")
    df_partition = df_partition.drop("element_id", axis=1)

    df_partition = df_partition.reset_index(drop=True)

    nodes = list(df_partition.node)
    communities = list(df_partition.community)

    for node, community_id in zip(nodes, communities):
        # for node, community_id in partition.items():
        g.nodes[node]["community"] = community_id

    df_partition["community"] = df_partition["community"].astype(int)

    # Add the nodes
    df_nodes["size"] = pd.qcut(
        df_nodes["sum_weight"].rank(method="first"),
        node_bins,
        np.arange(1, node_bins + 1),
    ).astype(int)

    scaler = MinMaxScaler(feature_range=(1, node_bins))
    df_nodes["size"] = scaler.fit_transform(df_nodes[["size"]])

    for _, row in df_nodes.iterrows():
        node_id = row["node"]
        node_size = int(row["size"])

        if node_id in list(g.nodes):
            g.add_node(node_id, node_size=node_size)

    # g = nx.DiGraph(g)

    Sigma.write_html(
        g,
        filepath,
        # node_size=g.degree,
        raw_node_size="node_size",
        node_size="node_size",
        fullscreen=True,
        # node_color="community",
        edge_size="weight",
        # node_label_color="community",
        node_label_color_palette="Dark2",
        node_color_palette="Dark2",
        start_layout=True,
        show_all_labels=True,
        # edge_color_palette="Dark2",
        # edge_color="community",
        # edge_color_from="source",
        # default_node_halo_color = 'white',
        # node_halo_color = 'white',
        # raw_node_halo_color = 'white',
        edge_size_range=(1, 5),
        node_size_range=(3, 20),
        # max_categorical_colors=1,
        max_categorical_colors=len(set(df_partition["community"])),
        default_edge_type="curve",
        node_border_color_from="node",
        default_node_label_size=25,
        # node_label_size=g.degree,
        # node_label_size="node_size",
        node_label_size_range=(7, 20),
    )

    return df_partition, g


def sygma_graph(
    df_edges: pd.DataFrame,
    df_nodes: pd.DataFrame,
    filepath: str,
    edge_bins: int = 10,
    node_bins: int = 10,
    resolution: int = 5,
):
    df_edges["weight"] = pd.qcut(
        df_edges["weight"].rank(method="first"), edge_bins, np.arange(1, edge_bins + 1)
    ).astype(int)
    df_edges = df_edges[["source", "target", "weight"]].reset_index(drop=True)

    # Create Graph Object
    g = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", edge_attr="weight"
    )

    # get the clusters and add it to the graph object
    partition = community.best_partition(g, resolution=resolution, random_state=42)

    # Randomy change the colors
    # from functions.feat_utils import shuffle_numbers_dict

    # shuffled_dict = shuffle_numbers_dict(list(set(partition.values())))
    # partition = {key: shuffled_dict[value] for key, value in partition.items()}

    for node, community_id in partition.items():
        g.nodes[node]["community"] = community_id

    df_partition = pd.DataFrame(partition, index=[0]).T.reset_index()
    df_partition.columns = ["node", "community"]
    df_partition = df_partition.sort_values("community")
    df_partition["community"] = df_partition["community"].astype(int)

    # Add the nodes
    df_nodes["size"] = pd.qcut(
        df_nodes["sum_weight"].rank(method="first"),
        node_bins,
        np.arange(1, node_bins + 1),
    ).astype(int)

    scaler = MinMaxScaler(feature_range=(1, node_bins))
    df_nodes["size"] = scaler.fit_transform(df_nodes[["size"]])

    for _, row in df_nodes.iterrows():
        node_id = row["node"]
        node_size = int(row["size"])

        if node_id in list(g.nodes):
            g.add_node(node_id, node_size=node_size)

    # g = nx.DiGraph(g)

    Sigma.write_html(
        g,
        filepath,
        # node_size=g.degree,
        raw_node_size="node_size",
        node_size="node_size",
        fullscreen=True,
        node_color="community",
        edge_size="weight",
        node_label_color="community",
        node_label_color_palette="Dark2",
        node_color_palette="Dark2",
        start_layout=True,
        show_all_labels=True,
        # edge_color_palette="Dark2",
        # edge_color="community",
        # edge_color_from="source",
        # default_node_halo_color = 'white',
        # node_halo_color = 'white',
        # raw_node_halo_color = 'white',
        edge_size_range=(1, 5),
        node_size_range=(3, 20),
        max_categorical_colors=len(set(df_partition["community"])),
        default_edge_type="curve",
        node_border_color_from="node",
        default_node_label_size=25,
        # node_label_size=g.degree,
        # node_label_size="node_size",
        node_label_size_range=(7, 20),
    )

    return df_partition, g
