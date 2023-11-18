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

# optimal_parameters = pd.read_sql("SELECT * FROM optimization_100", conn)
optimal_parameters = pd.read_sql("SELECT * FROM optimization_non_europe", conn)
optimal_parameters = optimal_parameters.sort_values("mean", ascending=False)

dict_op = optimal_parameters.iloc[0].to_dict()
dict_op = OptimumParameter(**dict_op)

from optimal_clustering import optimal_clustering

dict_op = optimal_clustering
dict_op = OptimumParameter(**dict_op)

from region_filters import columns_non_eu

if __name__ == "__main__":
    df_ind_regions = pd.read_sql_query(
        "SELECT * FROM individuals_regions", conn_full_db
    )
    df_ind_regions = df_ind_regions.rename(
        columns={"individual_wikidata_id": "wikidata_id"}
    )

    df_ind_regions = df_ind_regions[df_ind_regions["region_code"].isin(columns_non_eu)]

    df_occupation = pd.read_sql("SELECT * FROM individual_id_cleaned_occupations", conn)
    df_occupation.columns = ["source", "target"]
    df_occupation["weight"] = 1

    wiki_ids = list(set(df_ind_regions["wikidata_id"]))
    df = df_occupation[df_occupation["source"].isin(wiki_ids)]
    df = df.drop_duplicates()
    print(len(df))

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
        edge_bins=5,
        node_bins=10,
        filepath=GRAPH_RESULTS + "/non_europe.html",
    )

    df_partition.to_sql("partition_non_europe", conn, if_exists="replace", index=False)

    print(df_partition)
    print(len(df_partition))
