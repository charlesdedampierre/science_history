import sys

sys.path.append("../")

import sqlite3

import pandas as pd
import polars as pl

from functions.datamodel import OptimumParameter
from functions.env import DB_SCIENCE_PATH, FULL_DB_PATH, GRAPH_RESULTS
from functions.feat_network import filter_edge_table, get_edge_node_table
from functions.feat_visualization import sygma_graph, sygma_graph_leiden

conn_full_db = sqlite3.connect(FULL_DB_PATH)
conn = sqlite3.connect(DB_SCIENCE_PATH)

from optimal_clustering import optimal_clustering

dict_op = optimal_clustering
dict_op = OptimumParameter(**dict_op)

from region_filters import (
    columns_eu,
    columns_non_eu,
    columns_non_eu_unique,
    columns_eu_unique,
)

columns_to_keep = columns_non_eu_unique + columns_eu_unique

if __name__ == "__main__":
    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)

    df_temporal = pd.read_sql("SELECT * FROM temporal_data", conn)
    df_temporal = df_temporal[df_temporal["region_code"].isin(columns_non_eu_unique)]
    df_temporal = df_temporal[["wikidata_id", "birthyear"]]
    df_temporal = df_temporal[df_temporal["birthyear"] <= 1700]
    print(len(set(df_temporal.wikidata_id)))

    df = pd.merge(df_occupation, df_temporal, on="wikidata_id")
    df = df.drop("birthyear", axis=1)
    df = df.drop_duplicates()

    df.columns = ["source", "target"]
    df["weight"] = 1

    df = pl.from_pandas(df)
    df_edge, df_nodes = get_edge_node_table(df)

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
        filepath=GRAPH_RESULTS + "/before_non_europe_1700.html",
    )

    df_partition.to_sql(
        "partition_before_non_europe_1700", conn, if_exists="replace", index=False
    )
