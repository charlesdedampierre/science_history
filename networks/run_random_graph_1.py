import sys

sys.path.append("../")

import sqlite3

import pandas as pd
import polars as pl

from functions.datamodel import OptimumParameter
from functions.env import DB_SCIENCE_PATH, GRAPH_RESULTS, DB_SCIENCE_PATH_NEW
from functions.feat_network import get_edge_node_table
from functions.feat_visualization import sygma_graph_leiden

conn = sqlite3.connect(DB_SCIENCE_PATH_NEW)

from optimal_clustering import optimal_clustering

dict_op = optimal_clustering
dict_op = OptimumParameter(**dict_op)

if __name__ == "__main__":
    df = pd.read_csv("data/random_1.csv")
    df["meta_occupation"] = df["meta_occupation"].apply(lambda x: x.split(" | "))
    df = df.explode("meta_occupation")
    df = df[["wikidata_id", "meta_occupation"]]
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df.columns = ["source", "target"]
    df["weight"] = 1

    print(df)

    df = pl.from_pandas(df)
    df_edge, df_nodes = get_edge_node_table(df)
    df_edge.to_csv("matrix/random_1.csv")

    df_edge_filter = df_edge[df_edge["weight"] >= dict_op.min_count_link]
    df_edge_filter = df_edge_filter[
        df_edge_filter["source"] != df_edge_filter["target"]
    ]
    df_edge_filter = df_edge_filter[
        df_edge_filter["rank_count"] <= dict_op.n_neighbours
    ]

    df_partition, g = sygma_graph_leiden(
        df_edge_filter,
        df_nodes,
        edge_bins=10,
        node_bins=10,
        filepath=GRAPH_RESULTS + "/random_1.html",
    )

    df_partition = df_partition.sort_values("community")
    df_partition.to_sql("partition_random_1", conn, if_exists="replace", index=False)
